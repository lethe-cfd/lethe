/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2020 by the Lethe authors
 *
 * This file is part of the Lethe library
 *
 * The Lethe library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the Lethe distribution.
 *
 * ---------------------------------------------------------------------
 */

#include <core/manifolds.h>
#include <core/solutions_output.h>

#include <dem/data_containers.h>
#include <dem/dem.h>
#include <dem/explicit_euler_integrator.h>
#include <dem/find_contact_detection_step.h>
#include <dem/find_maximum_particle_size.h>
#include <dem/gear3_integrator.h>
#include <dem/input_parameter_inspection.h>
#include <dem/list_insertion.h>
#include <dem/non_uniform_insertion.h>
#include <dem/particle_wall_nonlinear_force.h>
#include <dem/post_processing.h>
#include <dem/print_initial_information.h>
#include <dem/read_checkpoint.h>
#include <dem/read_mesh.h>
#include <dem/set_particle_particle_contact_force_model.h>
#include <dem/set_particle_wall_contact_force_model.h>
#include <dem/uniform_insertion.h>
#include <dem/velocity_verlet_integrator.h>
#include <dem/write_checkpoint.h>

#include <deal.II/base/table_handler.h>

#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_out.h>

#include <sys/stat.h>

template <int dim>
DEMSolver<dim>::DEMSolver(DEMSolverParameters<dim> dem_parameters)
  : mpi_communicator(MPI_COMM_WORLD)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , pcout({std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0})
  , parameters(dem_parameters)
  , triangulation(this->mpi_communicator)
  , mapping(1)
  , particles_insertion_step(0)
  , contact_build_number(0)
  , computing_timer(this->mpi_communicator,
                    this->pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , particle_handler(triangulation, mapping, DEM::get_number_properties())
  , contact_detection_step(true)
  , load_balance_step(true)
  , checkpoint_step(true)
  , contact_detection_frequency(
      parameters.model_parameters.contact_detection_frequency)
  , insertion_frequency(parameters.insertion_info.insertion_frequency)
  , standard_deviation_multiplier(2.5)
  , background_dh(triangulation)
  , has_floating_mesh(false)
{
  // Print simulation starting information
  print_initial_information(pcout, n_mpi_processes);

  // Check if the output directory exists
  std::string output_dir_name = parameters.simulation_control.output_folder;
  struct stat buffer;

  // If output directory does not exist, create it
  if (this_mpi_process == 0)
    {
      if (stat(output_dir_name.c_str(), &buffer) != 0)
        {
          create_output_folder(output_dir_name);
        }
    }

  // Change the behavior of the timer for situations when you don't want outputs
  if (parameters.timer.type == Parameters::Timer::Type::none)
    computing_timer.disable_output();
  simulation_control = std::make_shared<SimulationControlTransientDEM>(
    parameters.simulation_control);

  // In order to consider the particles when repartitioning the triangulation
  // the algorithm needs to know three things:
  //
  // 1. How much weight to assign to each cell (how many particles are in
  // there)
  // 2. How to pack the particles before shipping data around
  // 3. How to unpack the particles after repartitioning
  //
  // Attach the correct functions to the signals inside
  // parallel::distributed::Triangulation, which will be called every time the
  // repartition() or refinement functions are called.
  // These connections only need to be created once, so we might as well
  // have set them up in the constructor of this class, but for the purpose
  // of this example we want to group the particle related instructions.

  triangulation.signals.weight.connect(
    [](const typename Triangulation<dim>::cell_iterator &,
       const typename Triangulation<dim>::CellStatus) -> unsigned int {
      return 1000;
    });

  triangulation.signals.weight.connect(
    [&](const typename parallel::distributed::Triangulation<dim>::cell_iterator
          &cell,
        const typename parallel::distributed::Triangulation<dim>::CellStatus
          status) -> unsigned int { return this->cell_weight(cell, status); });

  // Setting contact detection method (constant or dynamic)
  if (parameters.model_parameters.contact_detection_method ==
      Parameters::Lagrangian::ModelParameters::ContactDetectionMethod::constant)
    {
      check_contact_search_step =
        &DEMSolver<dim>::check_contact_search_step_constant;
    }
  else if (parameters.model_parameters.contact_detection_method ==
           Parameters::Lagrangian::ModelParameters::ContactDetectionMethod::
             dynamic)
    {
      check_contact_search_step =
        &DEMSolver<dim>::check_contact_search_step_dynamic;
    }
  else
    {
      throw std::runtime_error(
        "Specified contact detection method is not valid");
    }

  // Setting load-balance method (single-step, frequent or dynamic)
  if (parameters.model_parameters.load_balance_method ==
      Parameters::Lagrangian::ModelParameters::LoadBalanceMethod::once)
    {
      check_load_balance_step = &DEMSolver<dim>::check_load_balance_once;
    }
  else if (parameters.model_parameters.load_balance_method ==
           Parameters::Lagrangian::ModelParameters::LoadBalanceMethod::frequent)
    {
      check_load_balance_step = &DEMSolver<dim>::check_load_balance_frequent;
    }
  else if (parameters.model_parameters.load_balance_method ==
           Parameters::Lagrangian::ModelParameters::LoadBalanceMethod::dynamic)
    {
      check_load_balance_step = &DEMSolver<dim>::check_load_balance_dynamic;
    }
  else if (parameters.model_parameters.load_balance_method ==
           Parameters::Lagrangian::ModelParameters::LoadBalanceMethod::none)
    {
      check_load_balance_step = &DEMSolver<dim>::no_load_balance;
    }
  else
    {
      throw std::runtime_error("Specified load balance method is not valid");
    }

  // Calling input_parameter_inspection to evaluate input parameters in the
  // parameter handler file, finding maximum particle diameter used in
  // polydisperse systems
  maximum_particle_diameter =
    find_maximum_particle_size(parameters.lagrangian_physical_properties,
                               standard_deviation_multiplier);
  neighborhood_threshold_squared =
    std::pow(parameters.model_parameters.neighborhood_threshold *
               maximum_particle_diameter,
             2);

  if (this_mpi_process == 0)
    input_parameter_inspection(parameters,
                               pcout,
                               standard_deviation_multiplier);

  grid_motion_object =
    std::make_shared<GridMotion<dim, dim>>(parameters.grid_motion,
                                           simulation_control->get_time_step());

  for (unsigned int i_solid = 0;
       i_solid < parameters.solid_objects->number_solids;
       ++i_solid)
    {
      solids.push_back(std::make_shared<SerialSolid<dim - 1, dim>>(
        this->parameters.solid_objects->solids[i_solid], i_solid));
    }

  // Generate solid objects
  container_manager.floating_mesh_info.resize(solids.size());

  // Resize particle_floating_mesh_in_contact
  if (solids.size() > 0)
    {
      has_floating_mesh = true;
      container_manager.particle_floating_mesh_in_contact.resize(solids.size());
    }
}

template <int dim>
unsigned int
DEMSolver<dim>::cell_weight(
  const typename parallel::distributed::Triangulation<dim>::cell_iterator &cell,
  const typename parallel::distributed::Triangulation<dim>::CellStatus status)
  const
{
  // Assign no weight to cells we do not own.
  if (!cell->is_locally_owned())
    return 0;

  // This determines how important particle work is compared to cell
  // work (by default every cell has a weight of 1000).
  // We set the weight per particle much higher to indicate that
  // the particle load is the only one that is important to distribute
  // in this example. The optimal value of this number depends on the
  // application and can range from 0 (cheap particle operations,
  // expensive cell operations) to much larger than 1000 (expensive
  // particle operations, cheap cell operations, like in this case).
  // This parameter will need to be tuned for the case of DEM.
  const unsigned int particle_weight =
    parameters.model_parameters.load_balance_particle_weight;

  switch (status)
    {
      case parallel::distributed::Triangulation<dim>::CELL_PERSIST:
      case parallel::distributed::Triangulation<dim>::CELL_REFINE:
        {
          const unsigned int n_particles_in_cell =
            particle_handler.n_particles_in_cell(cell);
          return n_particles_in_cell * particle_weight;
          break;
        }

      case parallel::distributed::Triangulation<dim>::CELL_INVALID:
        break;

      case parallel::distributed::Triangulation<dim>::CELL_COARSEN:
        {
          unsigned int n_particles_in_cell = 0;

          for (unsigned int child_index = 0;
               child_index < GeometryInfo<dim>::max_children_per_cell;
               ++child_index)
            n_particles_in_cell +=
              particle_handler.n_particles_in_cell(cell->child(child_index));

          return n_particles_in_cell * particle_weight;
        }
        break;

      default:
        Assert(false, ExcInternalError());
        break;
    }

  return 0;
}

template <int dim>
void
DEMSolver<dim>::setup_background_dofs()
{
  FE_Q<dim> background_fe(1);
  background_dh.distribute_dofs(background_fe);
}

template <int dim>
void
DEMSolver<dim>::load_balance()
{
  // Prepare particle handler for the adaptation of the triangulation to the
  // load
  particle_handler.prepare_for_coarsening_and_refinement();

  pcout << "-->Repartitionning triangulation" << std::endl;
  triangulation.repartition();

  // Unpack the particle handler after the mesh has been repartitioned
  particle_handler.unpack_after_coarsening_and_refinement();

  // Update neighbors of cells after load balance
  container_manager.update_cell_neighbors(triangulation, has_floating_mesh);


  // Update the container with all the combinations of background and
  // solid cells
  container_manager.store_floating_mesh_info(triangulation, solids);

  if (parameters.boundary_conditions.BC_type ==
      Parameters::Lagrangian::BCDEM::BoundaryType::periodic)
    {
      periodic_boundaries_object.map_periodic_cells(triangulation);
    }

  boundary_cell_object.build(
    triangulation,
    parameters.floating_walls,
    parameters.boundary_conditions.outlet_boundaries,
    parameters.mesh.check_for_diamond_cells,
    parameters.mesh.expand_particle_wall_contact_search,
    pcout);

  if (parameters.grid_motion.motion_type !=
      Parameters::Lagrangian::GridMotion<dim>::MotionType::none)
    boundary_cell_object.update_boundary_info_after_grid_motion(
      container_manager.updated_boundary_points_and_normal_vectors);

  const auto average_minimum_maximum_cells =
    Utilities::MPI::min_max_avg(triangulation.n_active_cells(),
                                mpi_communicator);

  const auto average_minimum_maximum_particles =
    Utilities::MPI::min_max_avg(particle_handler.n_locally_owned_particles(),
                                mpi_communicator);

  pcout << "Load balance finished " << std::endl;
  pcout
    << "Average, minimum and maximum number of particles on the processors are "
    << average_minimum_maximum_particles.avg << " , "
    << average_minimum_maximum_particles.min << " and "
    << average_minimum_maximum_particles.max << std::endl;
  pcout << "Minimum and maximum number of cells owned by the processors are "
        << average_minimum_maximum_cells.min << " and "
        << average_minimum_maximum_cells.max << std::endl;
}

template <int dim>
inline bool
DEMSolver<dim>::no_load_balance()
{
  return false;
}

template <int dim>
inline bool
DEMSolver<dim>::check_load_balance_once()
{
  bool load_balance_step = (simulation_control->get_step_number() ==
                            parameters.model_parameters.load_balance_step);

  if (load_balance_step || checkpoint_step)
    load_balance();

  return load_balance_step;
}

template <int dim>
inline bool
DEMSolver<dim>::check_load_balance_frequent()
{
  bool load_balance_step =
    (simulation_control->get_step_number() %
       parameters.model_parameters.load_balance_frequency ==
     0);

  if (load_balance_step || checkpoint_step)
    load_balance();

  return load_balance_step;
}

template <int dim>
inline bool
DEMSolver<dim>::check_load_balance_dynamic()
{
  bool load_balance_step = false;
  if (simulation_control->get_step_number() %
        parameters.model_parameters.dynamic_load_balance_check_frequency ==
      0)
    {
      unsigned int maximum_particle_number_on_proc = 0;
      unsigned int minimum_particle_number_on_proc = 0;

      maximum_particle_number_on_proc =
        Utilities::MPI::max(particle_handler.n_locally_owned_particles(),
                            mpi_communicator);
      minimum_particle_number_on_proc =
        Utilities::MPI::min(particle_handler.n_locally_owned_particles(),
                            mpi_communicator);

      if ((maximum_particle_number_on_proc - minimum_particle_number_on_proc) >
            parameters.model_parameters.load_balance_threshold *
              (particle_handler.n_global_particles() / n_mpi_processes) ||
          checkpoint_step)
        {
          load_balance();
          load_balance_step = true;
        }
    }


  return load_balance_step;
}

template <int dim>
inline bool
DEMSolver<dim>::check_contact_search_step_dynamic()
{
  bool sorting_in_subdomains_step =
    (particles_insertion_step || load_balance_step || contact_detection_step);

  contact_detection_step =
    find_contact_detection_step<dim>(particle_handler,
                                     simulation_control->get_time_step(),
                                     smallest_contact_search_criterion,
                                     mpi_communicator,
                                     sorting_in_subdomains_step,
                                     displacement);

  return contact_detection_step;
}

template <int dim>
inline bool
DEMSolver<dim>::check_contact_search_step_constant()
{
  return (
    (simulation_control->get_step_number() % contact_detection_frequency) == 0);
}

template <int dim>
bool
DEMSolver<dim>::insert_particles()
{
  if ((simulation_control->get_step_number() % insertion_frequency) == 1)
    {
      insertion_object->insert(particle_handler, triangulation, parameters);
      return true;
    }
  return false;
}

template <int dim>
void
DEMSolver<dim>::update_moment_of_inertia(
  dealii::Particles::ParticleHandler<dim> &particle_handler,
  std::vector<double> &                    MOI)
{
  MOI.resize(torque.size());

  for (auto &particle : particle_handler)
    {
      auto &particle_properties = particle.get_properties();
      MOI[particle.get_local_index()] =
        0.1 * particle_properties[DEM::PropertiesIndex::mass] *
        particle_properties[DEM::PropertiesIndex::dp] *
        particle_properties[DEM::PropertiesIndex::dp];
    }
}

template <int dim>
void
DEMSolver<dim>::particle_wall_contact_force()
{
  // Particle-wall contact force
  particle_wall_contact_force_object->calculate_particle_wall_contact_force(
    container_manager.particle_wall_in_contact,
    simulation_control->get_time_step(),
    torque,
    force);

  if (parameters.forces_torques.calculate_force_torque)
    {
      container_manager
        .forces_boundary_information[simulation_control->get_step_number()] =
        particle_wall_contact_force_object->get_force();
      container_manager
        .torques_boundary_information[simulation_control->get_step_number()] =
        particle_wall_contact_force_object->get_torque();
    }

  // Particle-floating wall contact force
  if (parameters.floating_walls.floating_walls_number > 0)
    {
      particle_wall_contact_force_object->calculate_particle_wall_contact_force(
        container_manager.particle_floating_wall_in_contact,
        simulation_control->get_time_step(),
        torque,
        force);
    }

  // Particle - floating mesh contact force
  if (has_floating_mesh)
    {
      particle_wall_contact_force_object
        ->calculate_particle_floating_wall_contact_force(
          container_manager.particle_floating_mesh_in_contact,
          simulation_control->get_time_step(),
          torque,
          force,
          solids);
    }

  particle_point_line_contact_force_object
    .calculate_particle_point_contact_force(
      &container_manager.particle_points_in_contact,
      parameters.lagrangian_physical_properties,
      force);

  if (dim == 3)
    {
      particle_point_line_contact_force_object
        .calculate_particle_line_contact_force(
          &container_manager.particle_lines_in_contact,
          parameters.lagrangian_physical_properties,
          force);
    }
}

template <int dim>
void
DEMSolver<dim>::finish_simulation()
{
  // Timer output
  if (parameters.timer.type == Parameters::Timer::Type::end)
    {
      this->computing_timer.print_summary();
    }

  // Testing
  if (parameters.test.enabled)
    {
      visualization_object.print_xyz(particle_handler, mpi_communicator, pcout);
    }

  // Outputting force and torques over boundary
  if (parameters.forces_torques.calculate_force_torque)
    {
      write_forces_torques_output_results<dim>(
        parameters.forces_torques.force_torque_output_name,
        parameters.forces_torques.output_frequency,
        triangulation.get_boundary_ids(),
        simulation_control->get_time_step(),
        container_manager.forces_boundary_information,
        container_manager.torques_boundary_information);
    }
}

template <int dim>
std::shared_ptr<Insertion<dim>>
DEMSolver<dim>::set_insertion_type(const DEMSolverParameters<dim> &parameters)
{
  if (parameters.insertion_info.insertion_method ==
      Parameters::Lagrangian::InsertionInfo::InsertionMethod::uniform)
    {
      insertion_object =
        std::make_shared<UniformInsertion<dim>>(parameters,
                                                maximum_particle_diameter);
    }
  else if (parameters.insertion_info.insertion_method ==
           Parameters::Lagrangian::InsertionInfo::InsertionMethod::non_uniform)
    {
      insertion_object =
        std::make_shared<NonUniformInsertion<dim>>(parameters,
                                                   maximum_particle_diameter);
    }
  else if (parameters.insertion_info.insertion_method ==
           Parameters::Lagrangian::InsertionInfo::InsertionMethod::list)
    {
      insertion_object = std::make_shared<ListInsertion<dim>>(parameters);
    }
  else
    {
      throw "The chosen insertion method is invalid";
    }
  return insertion_object;
}

template <int dim>
std::shared_ptr<Integrator<dim>>
DEMSolver<dim>::set_integrator_type(const DEMSolverParameters<dim> &parameters)
{
  if (parameters.model_parameters.integration_method ==
      Parameters::Lagrangian::ModelParameters::IntegrationMethod::
        velocity_verlet)
    {
      integrator_object = std::make_shared<VelocityVerletIntegrator<dim>>();
    }
  else if (parameters.model_parameters.integration_method ==
           Parameters::Lagrangian::ModelParameters::IntegrationMethod::
             explicit_euler)
    {
      integrator_object = std::make_shared<ExplicitEulerIntegrator<dim>>();
    }
  else if (parameters.model_parameters.integration_method ==
           Parameters::Lagrangian::ModelParameters::IntegrationMethod::gear3)
    {
      integrator_object = std::make_shared<Gear3Integrator<dim>>();
    }
  else
    {
      throw "The chosen integration method is invalid";
    }
  return integrator_object;
}

template <int dim>
void
DEMSolver<dim>::write_output_results()
{
  const std::string folder = parameters.simulation_control.output_folder;
  const std::string particles_solution_name =
    parameters.simulation_control.output_name;
  const unsigned int iter        = simulation_control->get_step_number();
  const double       time        = simulation_control->get_current_time();
  const unsigned int group_files = parameters.simulation_control.group_files;

  // Write particles
  Visualization<dim> particle_data_out;
  particle_data_out.build_patches(particle_handler,
                                  properties_class.get_properties_name());

  write_vtu_and_pvd<0, dim>(particles_pvdhandler,
                            particle_data_out,
                            folder,
                            particles_solution_name,
                            time,
                            iter,
                            group_files,
                            mpi_communicator);

  // Write background grid
  DataOut<dim> background_data_out;

  background_data_out.attach_dof_handler(background_dh);

  // Attach the solution data to data_out object
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  background_data_out.add_data_vector(subdomain, "subdomain");

  const std::string grid_solution_name =
    parameters.simulation_control.output_name + "-grid";

  background_data_out.build_patches();

  write_vtu_and_pvd<dim>(grid_pvdhandler,
                         background_data_out,
                         folder,
                         grid_solution_name,
                         time,
                         iter,
                         group_files,
                         mpi_communicator);

  if (simulation_control->get_output_boundaries())
    {
      DataOutFaces<dim> data_out_faces;

      // Setup background dofs to initiate right sized boundary_id vector
      setup_background_dofs();
      Vector<float> boundary_id(background_dh.n_dofs());

      // Attach the boundary_id to data_out_faces object
      BoundaryPostprocessor<dim> boundary;
      data_out_faces.attach_dof_handler(background_dh);
      data_out_faces.add_data_vector(boundary_id, boundary);
      data_out_faces.build_patches();

      write_boundaries_vtu<dim>(
        data_out_faces, folder, time, iter, this->mpi_communicator);
    }

  // Write all solid objects
  for (const auto &solid_object : solids)
    solid_object->write_output_results(simulation_control);
}

template <int dim>
void
DEMSolver<dim>::post_process_results()
{
  if (parameters.post_processing.calculate_particles_average_velocity)
    {
      post_processing_object.calculate_average_particles_velocity(
        triangulation, particle_handler);

      post_processing_object.write_average_particles_velocity(
        triangulation,
        grid_pvdhandler,
        parameters,
        simulation_control->get_current_time(),
        simulation_control->get_step_number(),
        mpi_communicator);
    }
  if (parameters.post_processing.calculate_granular_temperature)
    {
      post_processing_object.calculate_average_granular_temperature(
        triangulation, particle_handler);

      post_processing_object.write_granular_temperature(
        triangulation,
        grid_pvdhandler,
        parameters,
        simulation_control->get_current_time(),
        simulation_control->get_step_number(),
        mpi_communicator);
    }
}

template <int dim>
void
DEMSolver<dim>::report_statistics()
{
  // Update statistics on contact list
  double number_of_list_built_since_last_log =
    double(contact_build_number) - contact_list.total;
  contact_list.max =
    std::max(number_of_list_built_since_last_log, contact_list.max);
  contact_list.min =
    std::min(number_of_list_built_since_last_log, contact_list.min);
  contact_list.total += number_of_list_built_since_last_log;
  contact_list.average = contact_list.total /
                         (simulation_control->get_step_number()) *
                         simulation_control->get_log_frequency();

  // Calculate statistics on the particles
  statistics translational_kinetic_energy = calculate_granular_statistics<
    dim,
    DEM::dem_statistic_variable::translational_kinetic_energy>(
    particle_handler, mpi_communicator);
  statistics rotational_kinetic_energy = calculate_granular_statistics<
    dim,
    DEM::dem_statistic_variable::rotational_kinetic_energy>(particle_handler,
                                                            mpi_communicator);
  statistics velocity =
    calculate_granular_statistics<dim, DEM::dem_statistic_variable::velocity>(
      particle_handler, mpi_communicator);
  statistics omega =
    calculate_granular_statistics<dim, DEM::dem_statistic_variable::omega>(
      particle_handler, mpi_communicator);

  if (this_mpi_process == 0)
    {
      TableHandler report;

      report.declare_column("Variable");
      report.declare_column("Min");
      report.declare_column("Max");
      report.declare_column("Average");
      report.declare_column("Total");
      add_statistics_to_table_handler("Contact list generation",
                                      contact_list,
                                      report);
      add_statistics_to_table_handler("Velocity magnitude", velocity, report);
      add_statistics_to_table_handler("Angular velocity magnitude",
                                      omega,
                                      report);
      add_statistics_to_table_handler("Translational kinetic energy",
                                      translational_kinetic_energy,
                                      report);
      add_statistics_to_table_handler("Rotational kinetic energy",
                                      rotational_kinetic_energy,
                                      report);



      report.set_scientific("Min", true);
      report.set_scientific("Max", true);
      report.set_scientific("Average", true);
      report.set_scientific("Total", true);

      report.write_text(std::cout, dealii::TableHandler::org_mode_table);
    }

  // Timer output
  if (parameters.timer.type == Parameters::Timer::Type::iteration)
    {
      this->computing_timer.print_summary();
    }
}

template <int dim>
void
DEMSolver<dim>::solve()
{
  // Reading mesh
  read_mesh(parameters.mesh,
            parameters.restart.restart,
            pcout,
            triangulation,
            triangulation_cell_diameter,
            parameters.boundary_conditions);


  container_manager.store_floating_mesh_info(triangulation, solids);

  if (parameters.restart.restart == true)
    {
      read_checkpoint(computing_timer,
                      parameters,
                      simulation_control,
                      particles_pvdhandler,
                      triangulation,
                      particle_handler);

      displacement.resize(particle_handler.get_max_local_particle_index());
      force.resize(displacement.size());
      torque.resize(displacement.size());

      update_moment_of_inertia(particle_handler, MOI);

      checkpoint_step = true;
    }

  // Find the smallest contact search frequency criterion between (smallest
  // cell size - largest particle radius) and (security factor * (blab
  // diamater
  // - 1) *  largest particle radius). This value is used in
  // find_contact_detection_frequency function
  smallest_contact_search_criterion =
    std::min((GridTools::minimal_cell_diameter(triangulation) -
              maximum_particle_diameter * 0.5),
             (parameters.model_parameters.dynamic_contact_search_factor *
              (parameters.model_parameters.neighborhood_threshold - 1) *
              maximum_particle_diameter * 0.5));

  // Find cell neighbors
  container_manager.execute_cell_neighbors_search(triangulation,
                                                  has_floating_mesh);

  if (parameters.boundary_conditions.BC_type ==
      Parameters::Lagrangian::BCDEM::BoundaryType::periodic)
    {
      periodic_boundaries_object.set_periodic_boundaries_information(
        parameters.boundary_conditions.outlet_boundaries,
        parameters.boundary_conditions.periodic_boundaries,
        parameters.boundary_conditions.periodic_direction);

      periodic_boundaries_object.map_periodic_cells(triangulation);
    }

  // Finding boundary cells with faces
  boundary_cell_object.build(
    triangulation,
    parameters.floating_walls,
    parameters.boundary_conditions.outlet_boundaries,
    parameters.mesh.check_for_diamond_cells,
    parameters.mesh.expand_particle_wall_contact_search,
    pcout);

  // Setting chosen contact force, insertion and integration methods
  insertion_object  = set_insertion_type(parameters);
  integrator_object = set_integrator_type(parameters);
  particle_particle_contact_force_object =
    set_particle_particle_contact_force_model(parameters);
  particle_wall_contact_force_object =
    set_particle_wall_contact_force_model(parameters,
                                          triangulation,
                                          triangulation_cell_diameter);

  // DEM engine iterator:
  while (simulation_control->integrate())
    {
      simulation_control->print_progression(pcout);
      if (simulation_control->is_verbose_iteration())
        report_statistics();

      // Grid motion
      if (parameters.grid_motion.motion_type !=
          Parameters::Lagrangian::GridMotion<dim>::MotionType::none)
        {
          grid_motion_object->move_grid(triangulation);
          boundary_cell_object.update_boundary_info_after_grid_motion(
            container_manager.updated_boundary_points_and_normal_vectors);
        }

      // Keep track if particles were inserted this step
      particles_insertion_step = insert_particles();

      if (particles_insertion_step)
        displacement.resize(particle_handler.get_max_local_particle_index());

      // Load balancing
      load_balance_step = (this->*check_load_balance_step)();

      if (load_balance_step || checkpoint_step)
        displacement.resize(particle_handler.get_max_local_particle_index());

      // Check to see if it is contact search step
      contact_detection_step = (this->*check_contact_search_step)();

      // Sort particles in cells
      if (particles_insertion_step || load_balance_step ||
          contact_detection_step || checkpoint_step)
        {
          particle_handler.sort_particles_into_subdomains_and_cells();

          displacement.resize(particle_handler.get_max_local_particle_index());
          force.resize(displacement.size());
          torque.resize(displacement.size());

          // Updating moment of inertia container
          update_moment_of_inertia(particle_handler, MOI);

          particle_handler.exchange_ghost_particles(true);
        }
      else
        {
          particle_handler.update_ghost_particles();
        }

      // Modify particles contact containers by search sequence
      if (particles_insertion_step || load_balance_step ||
          contact_detection_step || checkpoint_step)
        {
          // Reset checkpoint step
          checkpoint_step = false;

          // Execute board search by filling containers of particle-particle
          // contact pair candidates
          container_manager.execute_particle_particle_broad_search(
            particle_handler);

          // Updating number of contact builds
          contact_build_number++;

          // Execute board search by filling containers of particle-wall
          // contact pair candidates
          container_manager.execute_particle_wall_broad_search(
            particle_handler,
            boundary_cell_object,
            parameters.floating_walls,
            simulation_control->get_current_time(),
            has_floating_mesh);

          // Localize contacts, remove replicates and add new contact pairs
          // to the contact containers when particles are exchanged between
          // processors
          container_manager.localize_contacts();

          // Updates the iterators to particles in local-local contact
          // containers
          container_manager.update_local_particles_in_cells(particle_handler);

          // Execute fine search by updating particle-particle contact
          // containers according to the neighborhood threshold
          container_manager.execute_particle_particle_fine_search(
            neighborhood_threshold_squared);

          // Execute fine search by updating particle-wall contact containers
          // according to the neighborhood threshold
          container_manager.execute_particle_wall_fine_search(
            parameters.floating_walls,
            simulation_control->get_current_time(),
            neighborhood_threshold_squared,
            has_floating_mesh);
        }

      // Particle-particle contact force
      particle_particle_contact_force_object
        ->calculate_particle_particle_contact_force(
          container_manager,
          simulation_control->get_time_step(),
          torque,
          force);

      // We have to update the positions of the points on boundary faces and
      // their normal vectors here. The localize_contact class deletes the
      // particle-wall contact candidate if it exists in the contact list. As a
      // result, when we update the points on boundary faces and their normal
      // vectors, localize_contact deletes it from the output of broad search
      // and they are not updated in the contact force calculations.
      if (parameters.grid_motion.motion_type !=
          Parameters::Lagrangian::GridMotion<dim>::MotionType::none)
        grid_motion_object
          ->update_boundary_points_and_normal_vectors_in_contact_list(
            container_manager.particle_wall_in_contact,
            container_manager.updated_boundary_points_and_normal_vectors);

      // Move the solid triangulations, previous time must be used here instead
      // of current time.
      for (auto &solid_object : solids)
        solid_object->move_solid_triangulation(
          simulation_control->get_time_step(),
          simulation_control->get_previous_time());

      // Particles-walls contact force:
      particle_wall_contact_force();

      // Integration correction step (after force calculation)
      // In the first step, we have to obtain location of particles at half-step
      // time
      if (simulation_control->get_step_number() == 0)
        {
          integrator_object->integrate_half_step_location(
            particle_handler,
            parameters.lagrangian_physical_properties.g,
            simulation_control->get_time_step(),
            torque,
            force,
            MOI);
        }
      else
        {
          integrator_object->integrate(
            particle_handler,
            parameters.lagrangian_physical_properties.g,
            simulation_control->get_time_step(),
            torque,
            force,
            MOI);
        }

      // Particles displacement if passing through a periodic boundary
      periodic_boundaries_object.execute_particles_displacement(
        particle_handler);

      // Visualization
      if (simulation_control->is_output_iteration())
        {
          write_output_results();
        }
      if (parameters.forces_torques.calculate_force_torque &&
          (this_mpi_process == 0) &&
          (simulation_control->get_step_number() %
             parameters.forces_torques.output_frequency ==
           0) &&
          (parameters.forces_torques.force_torque_verbosity ==
           Parameters::Verbosity::verbose))
        write_forces_torques_output_locally(
          container_manager
            .forces_boundary_information[simulation_control->get_step_number()],
          container_manager.torques_boundary_information
            [simulation_control->get_step_number()]);

      // Post-processing
      if (parameters.post_processing.Lagrangian_post_processing)
        {
          if (simulation_control->get_step_number() >=
                parameters.post_processing.initial_step &&
              simulation_control->get_step_number() <=
                parameters.post_processing.end_step)
            {
              if (simulation_control->get_step_number() %
                    parameters.post_processing.output_frequency ==
                  0)
                post_process_results();
            }
        }

      if (parameters.restart.checkpoint &&
          simulation_control->get_step_number() %
              parameters.restart.frequency ==
            0)
        {
          write_checkpoint(computing_timer,
                           parameters,
                           simulation_control,
                           particles_pvdhandler,
                           triangulation,
                           particle_handler,
                           pcout,
                           mpi_communicator);
        }
    }

  finish_simulation();
}

template class DEMSolver<2>;
template class DEMSolver<3>;

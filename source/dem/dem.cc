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

 *
 * Author: Bruno Blais, Shahab Golshan, Polytechnique Montreal, 2019-
 */

#include <dem/dem.h>

template <int dim>
DEMSolver<dim>::DEMSolver(DEMSolverParameters<dim> dem_parameters)
  : mpi_communicator(MPI_COMM_WORLD)
  , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
  , pcout({std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0})
  , parameters(dem_parameters)
  , triangulation(this->mpi_communicator)
  , property_pool(DEM::get_number_properties())
  , mapping(1)
  , computing_timer(this->mpi_communicator,
                    this->pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , particle_handler(triangulation, mapping, DEM::get_number_properties())
{}

template <int dim>
void
DEMSolver<dim>::read_mesh()
{
  // GMSH input
  if (parameters.mesh.type == Parameters::Mesh::Type::gmsh)
    {
      GridIn<dim> grid_in;
      grid_in.attach_triangulation(triangulation);
      std::ifstream input_file(parameters.mesh.file_name);
      grid_in.read_msh(input_file);
    }

  // Dealii grids
  else if (parameters.mesh.type == Parameters::Mesh::Type::dealii)
    {
      GridGenerator::generate_from_name_and_arguments(
        triangulation,
        parameters.mesh.grid_type,
        parameters.mesh.grid_arguments);
    }
  else
    throw std::runtime_error(
      "Unsupported mesh type - mesh will not be created");

  const int initialSize = parameters.mesh.initialRefinement;
  triangulation.refine_global(initialSize);
}

template <int dim>
void
DEMSolver<dim>::reinitialize_force(
  Particles::ParticleHandler<dim> &particle_handler)
{
  for (auto particle = particle_handler.begin();
       particle != particle_handler.end();
       ++particle)
    {
      // Getting properties of particle as local variable
      auto particle_properties = particle->get_properties();

      // Reinitializing forces and momentums of particles in the system
      particle_properties[DEM::PropertiesIndex::force_x] = 0;
      particle_properties[DEM::PropertiesIndex::force_y] = 0;
      particle_properties[DEM::PropertiesIndex::force_z] = 0;

      particle_properties[DEM::PropertiesIndex::M_x] = 0;
      particle_properties[DEM::PropertiesIndex::M_y] = 0;
      particle_properties[DEM::PropertiesIndex::M_z] = 0;
    }
}

template <int dim>
std::map<int, Particles::ParticleIterator<dim>>
DEMSolver<dim>::update_particle_container(
  const Particles::ParticleHandler<dim> &particle_handler)
{
  std::map<int, Particles::ParticleIterator<dim>> particle_container;
  for (auto particle_iterator = particle_handler.begin();
       particle_iterator != particle_handler.end();
       ++particle_iterator)
    {
      particle_container[particle_iterator->get_id()] = particle_iterator;
    }
  return particle_container;
}

template <int dim>
void
DEMSolver<dim>::update_pp_contact_container_iterators(
  std::map<int, std::map<int, pp_contact_info_struct<dim>>>
    &                                                    pairs_in_contact_info,
  const std::map<int, Particles::ParticleIterator<dim>> &particle_container)
{
  for (auto pp_pairs_in_contact_iterator = pairs_in_contact_info.begin();
       pp_pairs_in_contact_iterator != pairs_in_contact_info.end();
       ++pp_pairs_in_contact_iterator)
    {
      int  particle_one_id          = pp_pairs_in_contact_iterator->first;
      auto pairs_in_contant_content = &pp_pairs_in_contact_iterator->second;
      for (auto pp_map_iterator = pairs_in_contant_content->begin();
           pp_map_iterator != pairs_in_contant_content->end();
           ++pp_map_iterator)
        {
          int particle_two_id = pp_map_iterator->first;
          pp_map_iterator->second.particle_one =
            particle_container.at(particle_one_id);
          pp_map_iterator->second.particle_two =
            particle_container.at(particle_two_id);
        }
    }
}

template <int dim>
void
DEMSolver<dim>::update_pw_contact_container_iterators(
  std::map<int, std::map<int, pw_contact_info_struct<dim>>>
    &                                                    pw_pairs_in_contact,
  const std::map<int, Particles::ParticleIterator<dim>> &particle_container)
{
  for (auto pw_pairs_in_contact_iterator = pw_pairs_in_contact.begin();
       pw_pairs_in_contact_iterator != pw_pairs_in_contact.end();
       ++pw_pairs_in_contact_iterator)
    {
      int  particle_id              = pw_pairs_in_contact_iterator->first;
      auto pairs_in_contant_content = &pw_pairs_in_contact_iterator->second;
      for (auto pw_map_iterator = pairs_in_contant_content->begin();
           pw_map_iterator != pairs_in_contant_content->end();
           ++pw_map_iterator)
        {
          pw_map_iterator->second.particle = particle_container.at(particle_id);
        }
    }
}

template <int dim>
void
DEMSolver<dim>::solve()
{
  // Reading mesh
  read_mesh();

  // Initializing variables
  std::map<int, std::map<int, pp_contact_info_struct<dim>>>
                                                            pairs_in_contact_info;
  std::map<int, std::map<int, pw_contact_info_struct<dim>>> pw_pairs_in_contact;
  Tensor<1, dim>                                            g;
  if (dim == 3)
    {
      g[0] = parameters.physicalProperties.gx;
      g[1] = parameters.physicalProperties.gy;
      g[2] = parameters.physicalProperties.gz;
    }
  if (dim == 2)
    {
      g[0] = parameters.physicalProperties.gx;
      g[1] = parameters.physicalProperties.gy;
    }

  std::vector<std::pair<std::string, int>> properties =
    properties_class.get_properties_name();

  // Finding cell neighbors
  FindCellNeighbors<dim> cell_neighbors_object;
  cell_neighbor_list = cell_neighbors_object.find_cell_neighbors(triangulation);

  // Finding boundary cells
  FindBoundaryCellsInformation<dim> boundary_cell_object;
  boundary_cells_information =
    boundary_cell_object.find_boundary_cells_information(triangulation);

  // DEM engine iterator:
  while (DEM_step < number_of_steps)
    {
      // Moving walls

      // Insertion
      // Defining a bool variable to specify the insertion steps
      bool insertion_step = 0;
      computing_timer.enter_subsection("insertion");
      if (fmod(DEM_step, parameters.insertionInfo.insertion_frequency) == 1)
        {
          if (DEM_step < parameters.insertionInfo.insertion_steps_number)
            {
              // put this if inside the insertion class or use a local variable
              // instead of n_global_particles
              if (particle_handler.n_global_particles() <
                  parameters.simulationControl
                    .total_particle_number) // number < total number
                {
                  NonUniformInsertion<dim> ins2(parameters);
                  // UniformInsertion<dim> ins2(parameters);

                  ins2.insert(particle_handler,
                              triangulation,
                              property_pool,
                              parameters);
                  insertion_step = 1;
                }
            }
        }
      computing_timer.leave_subsection();

      // Sort particles in cells
      computing_timer.enter_subsection("sort_particles_in_cells");
      if (insertion_step == 1 ||
          DEM_step % parameters.model_parmeters.pp_broad_search_frequency ==
            0 ||
          DEM_step % parameters.model_parmeters.pw_broad_search_frequency == 0)
        {
          particle_handler.sort_particles_into_subdomains_and_cells();
          particle_container.clear();
          particle_container = update_particle_container(particle_handler);

          update_pp_contact_container_iterators(pairs_in_contact_info,
                                                particle_container);

          update_pw_contact_container_iterators(pw_pairs_in_contact,
                                                particle_container);
        }
      computing_timer.leave_subsection();

      // Force reinitilization
      computing_timer.enter_subsection("reinitialize_forces");
      reinitialize_force(particle_handler);
      computing_timer.leave_subsection();

      // PP contact search
      // PP broad search
      computing_timer.enter_subsection("pp_broad_search");
      if (insertion_step == 1 ||
          DEM_step % parameters.model_parmeters.pp_broad_search_frequency == 0)
        {
          pp_broad_search_object.find_PP_Contact_Pairs(particle_handler,
                                                       cell_neighbor_list,
                                                       contact_pair_candidates);
        }
      computing_timer.leave_subsection();

      // PP fine search
      computing_timer.enter_subsection("pp_fine_search");

      pp_fine_search_object.pp_Fine_Search(contact_pair_candidates,
                                           pairs_in_contact_info,
                                           parameters.simulationControl.dt);
      computing_timer.leave_subsection();

      // PP contact force
      computing_timer.enter_subsection("pp_contact_force");
      pp_force_object.calculate_pp_contact_force(&pairs_in_contact_info,
                                                 parameters);
      computing_timer.leave_subsection();

      // PW contact search
      // PW broad search
      computing_timer.enter_subsection("pw_broad_search");
      if (insertion_step == 1 ||
          DEM_step % parameters.model_parmeters.pw_broad_search_frequency == 0)
        {
          pw_broad_search_object.find_PW_Contact_Pairs(
            boundary_cells_information,
            particle_handler,
            pw_contact_candidates);
        }
      computing_timer.leave_subsection();

      // PW fine search
      computing_timer.enter_subsection("pw_fine_search");

      pw_fine_search_object.pw_Fine_Search(pw_contact_candidates,
                                           pw_pairs_in_contact,
                                           parameters.simulationControl.dt);

      computing_timer.leave_subsection();

      // PW contact force:
      computing_timer.enter_subsection("pw_contact_force");
      pw_force_object.calculate_pw_contact_force(&pw_pairs_in_contact,
                                                 parameters);
      computing_timer.leave_subsection();

      // Integration
      computing_timer.enter_subsection("integration");
      integrator_object.integrate(particle_handler,
                                  g,
                                  parameters.simulationControl.dt);
      computing_timer.leave_subsection();

      // Visualization
      computing_timer.enter_subsection("visualization");
      if (DEM_step % parameters.simulationControl.write_frequency == 0)
        {
          Visualization<dim> visObj;
          visObj.build_patches(particle_handler, properties);
          WriteVTU<dim> writObj;
          writObj.write_master_files(visObj, parameters);
          writObj.writeVTUFiles(visObj, DEM_step, DEM_time, parameters);
        }
      computing_timer.leave_subsection();

      // Print iteration

      if (fmod(DEM_step, parameters.model_parmeters.print_info_frequency) == 1)
        {
          std::cout << "Step " << DEM_step << std::endl;
          computing_timer.print_summary();
          std::cout
            << "-------------------------------------------------------------"
               "----------"
            << std::endl;
        }

      // std::cout << "number" << particle_handler.n_global_particles() <<
      // std::endl;

      // Update:
      DEM_step = DEM_step + 1;
      DEM_time = DEM_step * parameters.simulationControl.dt;
    }

  while (parameters.simulation_control.integrate())
    {
      printTime(this->pcout, parameters.simulation_control);
    }
}

template class DEMSolver<2>;
template class DEMSolver<3>;

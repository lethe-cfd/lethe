/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - by the Lethe authors
 *
 * This file is part of the Lethe library
 *
 * The Lethe library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 3.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the Lethe distribution.
 *
 * ---------------------------------------------------------------------*/

#include <core/bdf.h>
#include <core/grids.h>
#include <core/linear_solvers_and_preconditioners.h>
#include <core/manifolds.h>
#include <core/multiphysics.h>
#include <core/time_integration_utilities.h>
#include <core/utilities.h>

#include <solvers/mf_navier_stokes.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

template <int dim>
MFNavierStokesSolver<dim>::MFNavierStokesSolver(
  SimulationParameters<dim> &nsparam)
  : NavierStokesBase<dim, VectorType, IndexSet>(nsparam)
{
  AssertThrow(
    nsparam.fem_parameters.velocity_order ==
      nsparam.fem_parameters.pressure_order,
    dealii::ExcMessage(
      "Matrix free Navier-Stokes does not support different orders for the velocity and the pressure!"));

  this->fe = std::make_shared<FESystem<dim>>(
    FE_Q<dim>(nsparam.fem_parameters.velocity_order), dim + 1);

  if ((nsparam.stabilization.use_default_stabilization == true) ||
      nsparam.stabilization.stabilization ==
        Parameters::Stabilization::NavierStokesStabilization::pspg_supg)
    system_operator =
      std::make_shared<NavierStokesSUPGPSPGOperator<dim, double>>();
  else
    throw std::runtime_error(
      "Only SUPG/PSPG stabilization is supported at the moment.");
}

template <int dim>
MFNavierStokesSolver<dim>::~MFNavierStokesSolver()
{
  this->dof_handler.clear();
}

template <int dim>
void
MFNavierStokesSolver<dim>::solve()
{
  read_mesh_and_manifolds(
    *this->triangulation,
    this->simulation_parameters.mesh,
    this->simulation_parameters.manifolds_parameters,
    this->simulation_parameters.restart_parameters.restart,
    this->simulation_parameters.boundary_conditions);

  this->setup_dofs();
  this->set_initial_condition(
    this->simulation_parameters.initial_condition->type,
    this->simulation_parameters.restart_parameters.restart);
  this->update_multiphysics_time_average_solution();

  while (this->simulation_control->integrate())
    {
      this->forcing_function->set_time(
        this->simulation_control->get_current_time());

      if ((this->simulation_control->get_step_number() %
               this->simulation_parameters.mesh_adaptation.frequency !=
             0 ||
           this->simulation_parameters.mesh_adaptation.type ==
             Parameters::MeshAdaptation::Type::none ||
           this->simulation_control->is_at_start()) &&
          this->simulation_parameters.boundary_conditions.time_dependent)
        {
          update_boundary_conditions();
        }

      this->simulation_control->print_progression(this->pcout);
      this->dynamic_flow_control();

      if (this->simulation_control->is_at_start())
        {
          this->iterate();
        }
      else
        {
          NavierStokesBase<dim, VectorType, IndexSet>::refine_mesh();
          this->iterate();
        }

      this->postprocess(false);
      this->finish_time_step();
    }

  this->finish_simulation();
}

template <int dim>
void
MFNavierStokesSolver<dim>::setup_dofs_fd()
{
  TimerOutput::Scope t(this->computing_timer, "setup_dofs");

  // Clear matrix free operator
  this->system_operator->clear();

  // Fill the dof handler and initialize vectors
  this->dof_handler.distribute_dofs(*this->fe);

  if (this->simulation_parameters.linear_solver.preconditioner ==
      Parameters::LinearSolver::PreconditionerType::lsmg)
    this->dof_handler.distribute_mg_dofs();

  DoFRenumbering::Cuthill_McKee(this->dof_handler);

  this->locally_owned_dofs = this->dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);

  // Non Zero constraints
  define_non_zero_constraints();

  // Zero constraints
  define_zero_constraints();

  // Initialize matrix-free object
  unsigned int mg_level = numbers::invalid_unsigned_int;
  this->system_operator->reinit(
    *this->mapping,
    this->dof_handler,
    this->zero_constraints,
    *this->cell_quadrature,
    &(*this->forcing_function),
    this->simulation_parameters.physical_properties_manager
      .get_kinematic_viscosity_scale(),
    mg_level);


  // Initialize vectors using operator
  this->system_operator->initialize_dof_vector(this->present_solution);
  this->system_operator->initialize_dof_vector(this->evaluation_point);
  this->system_operator->initialize_dof_vector(this->newton_update);
  this->system_operator->initialize_dof_vector(this->system_rhs);
  this->system_operator->initialize_dof_vector(this->local_evaluation_point);

  // Initialize vectors of previous solutions
  for (auto &solution : this->previous_solutions)
    {
      this->system_operator->initialize_dof_vector(solution);
    }

  if (this->simulation_parameters.post_processing.calculate_average_velocities)
    {
      this->average_velocities->initialize_vectors(
        this->locally_owned_dofs,
        this->locally_relevant_dofs,
        this->fe->n_dofs_per_vertex(),
        this->mpi_communicator);

      if (this->simulation_parameters.restart_parameters.checkpoint)
        {
          this->average_velocities->initialize_checkpoint_vectors(
            this->locally_owned_dofs,
            this->locally_relevant_dofs,
            this->mpi_communicator);
        }
    }

  double global_volume =
    GridTools::volume(*this->triangulation, *this->mapping);

  this->pcout << "   Number of active cells:       "
              << this->triangulation->n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: "
              << this->dof_handler.n_dofs() << std::endl;
  this->pcout << "   Volume of triangulation:      " << global_volume
              << std::endl;
}

template <int dim>
void
MFNavierStokesSolver<dim>::update_boundary_conditions()
{
  double time = this->simulation_control->get_current_time();
  for (unsigned int i_bc = 0;
       i_bc < this->simulation_parameters.boundary_conditions.size;
       ++i_bc)
    {
      this->simulation_parameters.boundary_conditions.bcFunctions[i_bc]
        .u.set_time(time);
      this->simulation_parameters.boundary_conditions.bcFunctions[i_bc]
        .v.set_time(time);
      this->simulation_parameters.boundary_conditions.bcFunctions[i_bc]
        .w.set_time(time);
      this->simulation_parameters.boundary_conditions.bcPressureFunction[i_bc]
        .p.set_time(time);
    }
  define_non_zero_constraints();
  // Distribute constraints
  auto &nonzero_constraints = this->nonzero_constraints;
  nonzero_constraints.distribute(this->local_evaluation_point);
  this->present_solution = this->local_evaluation_point;
}

template <int dim>
void
MFNavierStokesSolver<dim>::set_initial_condition_fd(
  Parameters::InitialConditionType initial_condition_type,
  bool                             restart)
{
  if (restart)
    {
      this->pcout << "************************" << std::endl;
      this->pcout << "---> Simulation Restart " << std::endl;
      this->pcout << "************************" << std::endl;
      this->read_checkpoint();
    }
  else if (initial_condition_type == Parameters::InitialConditionType::nodal)
    {
      this->set_nodal_values();
      this->finish_time_step();
    }
  else
    {
      throw std::runtime_error(
        "Type of initial condition is not supported by MF Navier-Stokes");
    }
}

template <int dim>
void
MFNavierStokesSolver<dim>::assemble_system_matrix()
{
  // Required for compilation but not used for matrix free solvers.
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");
}

template <int dim>
void
MFNavierStokesSolver<dim>::assemble_system_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_operator->evaluate_residual(this->system_rhs,
                                           this->evaluation_point);

  this->system_rhs *= -1.0;
}

template <int dim>
void
MFNavierStokesSolver<dim>::update_multiphysics_time_average_solution()
{
  // TODO
}

template <int dim>
void
MFNavierStokesSolver<dim>::setup_preconditioner(SolverGMRES<VectorType> &solver)
{
  if (this->simulation_parameters.linear_solver.preconditioner ==
      Parameters::LinearSolver::PreconditionerType::lsmg)
    setup_LSMG(solver);
  else if (this->simulation_parameters.linear_solver.preconditioner ==
           Parameters::LinearSolver::PreconditionerType::gcmg)
    setup_GCMG();
}

template <int dim>
void
MFNavierStokesSolver<dim>::setup_LSMG(SolverGMRES<VectorType> &solver)
{
  // setup_ls_multigrid_preconditioner(
  //   this->ls_multigrid_preconditioner,
  //   this->simulation_parameters,
  //   *this->system_operator,
  //   *this->mapping,
  //   this->dof_handler,
  //   *this->cell_quadrature,
  //   this->present_solution,
  //   this->forcing_function,
  //   this->simulation_parameters.physical_properties_manager
  //     .get_viscosity_scale());
  using OperatorType               = NavierStokesSUPGPSPGOperator<dim, double>;
  using LSTransferType             = MGTransferMatrixFree<dim, double>;
  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType =
    PreconditionRelaxation<OperatorType, SmootherPreconditionerType>;
  using PreconditionerType = PreconditionMG<dim, VectorType, LSTransferType>;

  MGLevelObject<std::shared_ptr<OperatorType>> mg_operators;
  LSTransferType                               mg_transfer;
  MGLevelObject<VectorType>                    mg_solution;
  MGLevelObject<std::shared_ptr<OperatorType>> mg_interface_in;
  MGLevelObject<std::shared_ptr<OperatorType>> mg_interface_out;
  MGLevelObject<AffineConstraints<double>>     level_constraints;
  MGConstrainedDoFs                            mg_constrained_dofs;
  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<OperatorType>>
    ls_mg_operators;
  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<OperatorType>>
    ls_mg_interface_in;
  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<OperatorType>>
    ls_mg_interface_out;

  const unsigned int n_h_levels =
    this->dof_handler.get_triangulation().n_global_levels();

  unsigned int minlevel = 0;
  unsigned int maxlevel = n_h_levels - 1;

  mg_operators.resize(0, n_h_levels - 1);
  mg_solution.resize(0, n_h_levels - 1);
  level_constraints.resize(0, n_h_levels - 1);
  ls_mg_interface_in.resize(0, n_h_levels - 1);
  ls_mg_interface_out.resize(0, n_h_levels - 1);
  ls_mg_operators.resize(0, n_h_levels - 1);

  std::set<types::boundary_id> dirichlet_boundary_ids = {0, 1, 2, 3, 4, 5};

  mg_constrained_dofs.initialize(this->dof_handler);
  mg_constrained_dofs.make_zero_boundary_constraints(this->dof_handler,
                                                     dirichlet_boundary_ids);

  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners(
    this->dof_handler.get_triangulation().n_global_levels());

  for (unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      const IndexSet relevant_dofs =
        DoFTools::extract_locally_relevant_level_dofs(this->dof_handler, level);

      level_constraints[level].reinit(relevant_dofs);
      level_constraints[level].add_lines(
        mg_constrained_dofs.get_boundary_indices(level));
      level_constraints[level].close();

      mg_operators[level] =
        std::make_unique<NavierStokesSUPGPSPGOperator<dim, double>>();

      mg_operators[level]->reinit(
        *this->mapping,
        this->dof_handler,
        level_constraints[level],
        *this->cell_quadrature,
        this->forcing_function,
        this->simulation_parameters.physical_properties_manager
          .get_viscosity_scale(),
        level);

      mg_operators[level]->initialize_dof_vector(mg_solution[level]);

      ls_mg_operators[level].initialize(*mg_operators[level]);
      ls_mg_interface_in[level].initialize(*mg_operators[level]);
      ls_mg_interface_out[level].initialize(*mg_operators[level]);

      partitioners[level] = mg_operators[level]->get_vector_partitioner();
    }

  mg_transfer.initialize_constraints(mg_constrained_dofs);
  mg_transfer.build(this->dof_handler, partitioners);
  mg_transfer.interpolate_to_mg(this->dof_handler,
                                mg_solution,
                                this->present_solution);

  for (unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      mg_solution[level].update_ghost_values();
      mg_operators[level]->evaluate_non_linear_term(mg_solution[level]);
    }

  for (unsigned int level = minlevel; level <= maxlevel; ++level)
    this->pcout << "   MG Level " << level << ": "
                << this->dof_handler.n_dofs(level) << " DoFs, "
                << this->dof_handler.get_triangulation().n_cells(level)
                << " cells" << std::endl;

  mg::Matrix<VectorType> mg_matrix(ls_mg_operators);

  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(minlevel,
                                                                     maxlevel);

  for (unsigned int level = minlevel; level <= maxlevel; ++level)
    {
      smoother_data[level].preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>();
      mg_operators[level]->compute_inverse_diagonal(
        smoother_data[level].preconditioner->get_vector());
      smoother_data[level].n_iterations = 10;
      smoother_data[level].relaxation   = 0.5;
    }

  MGSmootherPrecondition<OperatorType, SmootherType, VectorType> mg_smoother;
  mg_smoother.initialize(mg_operators, smoother_data);

  ReductionControl coarse_grid_solver_control(2000, 1e-14, 1e-4, false, false);
  SolverGMRES<VectorType> coarse_grid_solver(coarse_grid_solver_control);

  std::shared_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  TrilinosWrappers::PreconditionAMG                 precondition_amg;
  TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
  amg_data.smoother_sweeps = 1;
  amg_data.n_cycles        = 1;
  amg_data.smoother_type   = "ILU";

  precondition_amg.initialize(mg_operators[minlevel]->get_system_matrix(),
                              amg_data);

  mg_coarse =
    std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                 SolverGMRES<VectorType>,
                                                 OperatorType,
                                                 decltype(precondition_amg)>>(
      coarse_grid_solver, *mg_operators[minlevel], precondition_amg);

  mg::Matrix<VectorType> mg_interface_matrix_in(ls_mg_interface_in);
  mg::Matrix<VectorType> mg_interface_matrix_out(ls_mg_interface_out);

  Multigrid<VectorType> mg(
    mg_matrix, *mg_coarse, mg_transfer, mg_smoother, mg_smoother);

  if (this->dof_handler.get_triangulation().has_hanging_nodes())
    mg.set_edge_matrices(mg_interface_matrix_in, mg_interface_matrix_out);

  ls_multigrid_preconditioner =
    std::make_shared<PreconditionMG<dim, VectorType, LSTransferType>>(
      this->dof_handler, mg, mg_transfer);

  solver.solve(*(system_operator),
               this->newton_update,
               this->system_rhs,
               *ls_multigrid_preconditioner);
}

template <int dim>
void
MFNavierStokesSolver<dim>::setup_GCMG()
{}

template <int dim>
void
MFNavierStokesSolver<dim>::define_non_zero_constraints()
{
  double time = this->simulation_control->get_current_time();
  FEValuesExtractors::Vector velocities(0);
  FEValuesExtractors::Scalar pressure(dim);
  // Non-zero constraints
  auto &nonzero_constraints = this->get_nonzero_constraints();
  {
    nonzero_constraints.clear();
    nonzero_constraints.reinit(this->locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(this->dof_handler,
                                            nonzero_constraints);
    for (unsigned int i_bc = 0;
         i_bc < this->simulation_parameters.boundary_conditions.size;
         ++i_bc)
      {
        if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
            BoundaryConditions::BoundaryType::noslip)
          {
            VectorTools::interpolate_boundary_values(
              *this->mapping,
              this->dof_handler,
              this->simulation_parameters.boundary_conditions.id[i_bc],
              dealii::Functions::ZeroFunction<dim>(dim + 1),
              nonzero_constraints,
              this->fe->component_mask(velocities));
          }
        else if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
                 BoundaryConditions::BoundaryType::slip)
          {
            std::set<types::boundary_id> no_normal_flux_boundaries;
            no_normal_flux_boundaries.insert(
              this->simulation_parameters.boundary_conditions.id[i_bc]);
            VectorTools::compute_no_normal_flux_constraints(
              this->dof_handler,
              0,
              no_normal_flux_boundaries,
              nonzero_constraints,
              *this->mapping);
          }
        else if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
                 BoundaryConditions::BoundaryType::function)
          {
            this->simulation_parameters.boundary_conditions.bcFunctions[i_bc]
              .u.set_time(time);
            this->simulation_parameters.boundary_conditions.bcFunctions[i_bc]
              .v.set_time(time);
            this->simulation_parameters.boundary_conditions.bcFunctions[i_bc]
              .w.set_time(time);
            VectorTools::interpolate_boundary_values(
              *this->mapping,
              this->dof_handler,
              this->simulation_parameters.boundary_conditions.id[i_bc],
              NavierStokesFunctionDefined<dim>(
                &this->simulation_parameters.boundary_conditions
                   .bcFunctions[i_bc]
                   .u,
                &this->simulation_parameters.boundary_conditions
                   .bcFunctions[i_bc]
                   .v,
                &this->simulation_parameters.boundary_conditions
                   .bcFunctions[i_bc]
                   .w),
              nonzero_constraints,
              this->fe->component_mask(velocities));
          }
        else if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
                 BoundaryConditions::BoundaryType::periodic)
          {
            DoFTools::make_periodicity_constraints(
              this->dof_handler,
              this->simulation_parameters.boundary_conditions.id[i_bc],
              this->simulation_parameters.boundary_conditions.periodic_id[i_bc],
              this->simulation_parameters.boundary_conditions
                .periodic_direction[i_bc],
              nonzero_constraints);
          }
      }
  }

  this->establish_solid_domain(true);

  nonzero_constraints.close();
}

template <int dim>
void
MFNavierStokesSolver<dim>::define_zero_constraints()
{
  FEValuesExtractors::Vector velocities(0);
  FEValuesExtractors::Scalar pressure(dim);
  this->zero_constraints.clear();
  DoFTools::extract_locally_relevant_dofs(this->dof_handler,
                                          this->locally_relevant_dofs);
  this->zero_constraints.reinit(this->locally_relevant_dofs);

  DoFTools::make_hanging_node_constraints(this->dof_handler,
                                          this->zero_constraints);

  for (unsigned int i_bc = 0;
       i_bc < this->simulation_parameters.boundary_conditions.size;
       ++i_bc)
    {
      if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
          BoundaryConditions::BoundaryType::slip)
        {
          std::set<types::boundary_id> no_normal_flux_boundaries;
          no_normal_flux_boundaries.insert(
            this->simulation_parameters.boundary_conditions.id[i_bc]);
          VectorTools::compute_no_normal_flux_constraints(
            this->dof_handler,
            0,
            no_normal_flux_boundaries,
            this->zero_constraints,
            *this->mapping);
        }
      else if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
               BoundaryConditions::BoundaryType::periodic)
        {
          DoFTools::make_periodicity_constraints(
            this->dof_handler,
            this->simulation_parameters.boundary_conditions.id[i_bc],
            this->simulation_parameters.boundary_conditions.periodic_id[i_bc],
            this->simulation_parameters.boundary_conditions
              .periodic_direction[i_bc],
            this->zero_constraints);
        }
      else if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
               BoundaryConditions::BoundaryType::pressure)
        {
          /*do nothing*/
        }
      else if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
               BoundaryConditions::BoundaryType::function_weak)
        {
          /*do nothing*/
        }
      else if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
               BoundaryConditions::BoundaryType::partial_slip)
        {
          /*do nothing*/
        }
      else if (this->simulation_parameters.boundary_conditions.type[i_bc] ==
               BoundaryConditions::BoundaryType::outlet)
        {
          /*do nothing*/
        }
      else
        {
          VectorTools::interpolate_boundary_values(
            *this->mapping,
            this->dof_handler,
            this->simulation_parameters.boundary_conditions.id[i_bc],
            dealii::Functions::ZeroFunction<dim>(dim + 1),
            this->zero_constraints,
            this->fe->component_mask(velocities));
        }
    }

  this->establish_solid_domain(false);

  this->zero_constraints.close();
}

template <int dim>
void
MFNavierStokesSolver<dim>::solve_linear_system(const bool initial_step,
                                               const bool /* renewed_matrix */)
{
  const double absolute_residual =
    this->simulation_parameters.linear_solver.at(PhysicsID::fluid_dynamics)
      .minimum_residual;
  const double relative_residual =
    this->simulation_parameters.linear_solver.at(PhysicsID::fluid_dynamics)
      .relative_residual;

  if (this->simulation_parameters.linear_solver.at(PhysicsID::fluid_dynamics)
        .solver == Parameters::LinearSolver::SolverType::gmres)
    solve_system_GMRES(initial_step, absolute_residual, relative_residual);
  else
    AssertThrow(false, ExcMessage("This solver is not allowed"));
  this->rescale_pressure_dofs_in_newton_update();
}

template <int dim>
void
MFNavierStokesSolver<dim>::assemble_L2_projection()
{
  // TODO
}

template <int dim>
void
MFNavierStokesSolver<dim>::solve_system_GMRES(const bool   initial_step,
                                              const double absolute_residual,
                                              const double relative_residual)
{
  auto &system_rhs          = this->system_rhs;
  auto &nonzero_constraints = this->nonzero_constraints;

  const AffineConstraints<double> &constraints_used =
    initial_step ? nonzero_constraints : this->zero_constraints;
  const double linear_solver_tolerance =
    std::max(relative_residual * system_rhs.l2_norm(), absolute_residual);

  if (this->simulation_parameters.linear_solver.at(PhysicsID::fluid_dynamics)
        .verbosity != Parameters::Verbosity::quiet)
    {
      this->pcout << "  -Tolerance of iterative solver is : "
                  << linear_solver_tolerance << std::endl;
    }

  SolverControl solver_control(this->simulation_parameters.linear_solver
                                 .at(PhysicsID::fluid_dynamics)
                                 .max_iterations,
                               linear_solver_tolerance,
                               true,
                               true);

  SolverGMRES<VectorType>::AdditionalData solver_parameters;

  solver_parameters.max_n_tmp_vectors =
    this->simulation_parameters.linear_solver.at(PhysicsID::fluid_dynamics)
      .max_krylov_vectors;

  // if (!ls_multigrid_preconditioner)
  //   setup_preconditioner();

  SolverGMRES<VectorType> solver(solver_control, solver_parameters);

          {
            TimerOutput::Scope t(this->computing_timer, "solve_linear_system");

            this->present_solution.update_ghost_values();

            this->system_operator->evaluate_non_linear_term(
              this->present_solution);

            this->newton_update = 0.0;

            if (this->simulation_parameters.linear_solver.preconditioner ==
                Parameters::LinearSolver::PreconditionerType::lsmg)
              setup_preconditioner(solver);
            // solver.solve(*(system_operator),
            //              this->newton_update,
            //              system_rhs,
            //              *ls_multigrid_preconditioner);
            else if (this->simulation_parameters.linear_solver.preconditioner ==
                     Parameters::LinearSolver::PreconditionerType::gcmg)
              solver.solve(*(system_operator),
                           this->newton_update,
                           system_rhs,
                           *gc_multigrid_preconditioner);
            else
              {
                // throw(std::runtime_error(
                //   "This solver with this preconditioner is not allowed"));
                PreconditionIdentity preconditioner;
                solver.solve(*(system_operator),
                             this->newton_update,
                             system_rhs,
                             preconditioner);
              }

            if (this->simulation_parameters.linear_solver
                  .at(PhysicsID::fluid_dynamics)
                  .verbosity != Parameters::Verbosity::quiet)
              {
                this->pcout
                  << "  -Iterative solver took : " << solver_control.last_step()
                  << " steps " << std::endl;
              }
          }

    if (this->simulation_parameters.linear_solver.verbosity !=
        Parameters::Verbosity::quiet)
      {
        this->pcout << "  -Iterative solver took : "
                    << solver_control.last_step() << " steps " << std::endl;
      }
  }

          if (iter == max_iter - 1 && !this->simulation_parameters.linear_solver
                                         .at(PhysicsID::fluid_dynamics)
                                         .force_linear_solver_continuation)
            throw e;
        }
      iter += 1;
    }
}

template class MFNavierStokesSolver<2>;
template class MFNavierStokesSolver<3>;

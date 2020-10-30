/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 -  by the Lethe authors
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
 * Author: Audrey Collard-Daigneault, Polytechnique Montreal, 2020 -
 */

#ifndef lethe_postprocessing_velocities_h
#define lethe_postprocessing_velocities_h

// Dealii Includes

// Base
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

// Lac
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

// Lac - Trilinos includes
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

// Grid
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

// Dofs
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

// Fe
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

// Numerics
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

// Distributed
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>


// Lethe Includes
#include <core/bdf.h>
#include <core/boundary_conditions.h>
#include <core/manifolds.h>
#include <core/newton_non_linear_solver.h>
#include <core/parameters.h>
#include <core/physics_solver.h>
#include <core/pvd_handler.h>
#include <core/simulation_control.h>
#include <solvers/flow_control.h>

#include "navier_stokes_solver_parameters.h"
#include "post_processors.h"

// Std
#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim, typename VectorType, typename DofsType>
class AverageVelocities
{
public:
  VectorType
  calculate_average_velocities(
    const VectorType &                   present_solution,
    const Parameters::SimulationControl &simulation_control,
    const Parameters::PostProcessing &   post_processing,
    const double &                       current_time,
    const DofsType &                     locally_owned_dofs,
    const DofsType &                     locally_relevant_dofs,
    const MPI_Comm &                     mpi_communicator);
  VectorType
  nondimensionalize_average_velocities(const double bulk_velocity);

private:
  VectorType sum_velocity_dt;
  VectorType average_velocities;
  VectorType nondimensionalized_average_velocities;
};

template <int dim, typename VectorType, typename DofsType>
VectorType
AverageVelocities<dim, VectorType, DofsType>::calculate_average_velocities(
  const VectorType &                   present_solution,
  const Parameters::SimulationControl &simulation_control,
  const Parameters::PostProcessing &   post_processing,
  const double &                       current_time,
  const DofsType &                     locally_owned_dofs,
  const DofsType &                     locally_relevant_dofs,
  const MPI_Comm &                     mpi_communicator)
{
  const double         total_time = current_time - post_processing.initial_time;
  const TrilinosScalar trilinos_total_time = total_time;
  const TrilinosScalar trilinos_dt         = simulation_control.dt;
  VectorType           velocity_dt;
  velocity_dt.reinit(locally_owned_dofs,
                     locally_relevant_dofs,
                     mpi_communicator);
  velocity_dt.equ(trilinos_dt, present_solution);

  if (current_time - 0.0 < 1e-6)
    {
      std::cout << "foo0" << std::endl;
      // Reinitilizing vectors with zeros at t = 0
      sum_velocity_dt.reinit(present_solution);
      average_velocities.reinit(present_solution);
      std::cout << "foo" << std::endl;
    }
  else if (abs(total_time) < 1e-6)
    {
      // Starting to sum velocity*dt because t = initial time
      sum_velocity_dt = velocity_dt;
      std::cout << "foo2" << std::endl;
    }
  else if (total_time >= 1e-6)
    {
      // Generating average velocities at each time step
      sum_velocity_dt += velocity_dt;
      average_velocities.equ(1. / trilinos_total_time, sum_velocity_dt);
      std::cout << "foo3" << std::endl;
    }
  std::cout << "foo4" << std::endl;
  return average_velocities;
}

template <int dim, typename VectorType, typename DofsType>
VectorType
AverageVelocities<dim, VectorType, DofsType>::
  nondimensionalize_average_velocities(const double bulk_velocity)
{
  const TrilinosScalar trilinos_bulk_velocity = bulk_velocity;
  nondimensionalized_average_velocities       = average_velocities;
  nondimensionalized_average_velocities.equ(trilinos_bulk_velocity,
                                            average_velocities);
  return nondimensionalized_average_velocities;
}


#endif

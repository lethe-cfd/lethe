/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 -  by the Lethe authors
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

#ifndef kinsol_non_linear_solver_h
#define kinsol_non_linear_solver_h

#include <core/multiphysics.h>
#include <core/non_linear_solver.h>

#include <deal.II/sundials/kinsol.h>

/**
 * @brief KinsolNonlinearSolver. Interface to the non-linear newton solver for non-linear systems
 * of equations implemented in the SUNDIALS suite, specifically the KINSOL
 * package. This solver has internal algorithms to determine the time step and
 * decide whether to reassemble the Jacobian matrix or not.
 */
template <typename VectorType>
class KinsolNewtonNonLinearSolver : public NonLinearSolver<VectorType>
{
public:
  /**
   * @brief Constructor for the KinsolNewtonNonLinearSolver.
   *
   * @param physics_solver A pointer to the physics solver to which the non-linear solver is attached
   *
   * @param param Non-linear solver parameters
   *
   */
  KinsolNewtonNonLinearSolver(PhysicsSolver<VectorType> *        physics_solver,
                              const Parameters::NonLinearSolver &param);


  /**
   * @brief Solve the non-linear system of equation.
   *
   * @param time_stepping_method Time stepping method being used. This is
   * required since the jacobian of the matrix is going to depend on the method
   * used
   *
   * @param is_initial_step Boolean variable that controls which constraints are
   * going to be applied to the equations
   */
  void
  solve(const Parameters::SimulationControl::TimeSteppingMethod
                   time_stepping_method,
        const bool is_initial_step) override;
};

template <typename VectorType>
KinsolNewtonNonLinearSolver<VectorType>::KinsolNewtonNonLinearSolver(
  PhysicsSolver<VectorType> *        physics_solver,
  const Parameters::NonLinearSolver &params)
  : NonLinearSolver<VectorType>(physics_solver, params)
{}

template <typename VectorType>
void
KinsolNewtonNonLinearSolver<VectorType>::solve(
  const Parameters::SimulationControl::TimeSteppingMethod time_stepping_method,
  const bool                                              is_initial_step)
{
  bool first_step = is_initial_step;

  PhysicsSolver<VectorType> *solver = this->physics_solver;

  VectorType &evaluation_point = solver->get_evaluation_point();
  VectorType &present_solution = solver->get_present_solution();

  typename SUNDIALS::KINSOL<VectorType>::AdditionalData additional_data;
  additional_data.function_tolerance = this->params.tolerance;
  additional_data.function_tolerance = this->params.max_iterations;
  SUNDIALS::KINSOL<VectorType> nonlinear_solver(additional_data);

  nonlinear_solver.reinit_vector = [&](VectorType &x) {
    x.reinit(present_solution);
  };

  nonlinear_solver.residual = [&](const VectorType &evaluation_point_for_kinsol,
                                  VectorType &      residual) {
    std::cout << "Computing residual vector..." << std::endl;

    evaluation_point = evaluation_point_for_kinsol;
    solver->apply_constraints();
    solver->assemble_rhs(time_stepping_method);
    auto &current_residual = solver->get_system_rhs();
    residual               = current_residual;
    return 0;
  };

  nonlinear_solver.setup_jacobian =
    [&](const VectorType &present_solution_for_kinsol,
        const VectorType & /*current_f*/) {
      // TODO Replace the function by assemble matrix only
      std::cout << "Computing jacobian matrix..." << std::endl;
      evaluation_point = present_solution_for_kinsol;
      solver->assemble_matrix_and_rhs(time_stepping_method);
      return 0;
    };

  nonlinear_solver.solve_with_jacobian = [&](const VectorType &residual,
                                             VectorType &      dst,
                                             const double /* tolerance */) {
    std::cout << "Solving linear system..." << std::endl;
    solver->solve_linear_system(first_step);
    dst = solver->get_newton_update();
    return 0;
  };

  nonlinear_solver.solve(present_solution);
}

#endif

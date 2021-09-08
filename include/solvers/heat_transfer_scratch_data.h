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
 * Implementation of heat transfer as an auxiliary physics.
 * This heat equation is weakly coupled to the velocity field.
 * Equation solved:
 * rho * Cp * (dT/dt + u.gradT) = k div(gradT) + nu/rho * (gradu : gradu)
 *
 * Author: Bruno Blais, Jeanne Joachim and Shahab Golshan, Polytechnique
 Montreal, 2020-
 */

#ifndef lethe_heat_transfer_scratch_data_h
#define lethe_heat_transfer_scratch_data_h

#include <core/multiphysics.h>

#include <solvers/multiphysics_interface.h>

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/numerics/vector_tools.h>

using namespace dealii;


/**
 * @brief TracerScratchData class
 * stores the information required by the assembly procedure
 * for a Tracer advection-diffusion equation. Consequently, this class
 *calculates the tracer (values, gradients, laplacians) and the shape function
 * (values, gradients, laplacians) at all the gauss points for all degrees
 * of freedom and stores it into arrays. Additionnaly, the use can request
 * that this class gathers additional fields for physics which are coupled
 * to the Tracer equation, such as the velocity which is required. This class
 * serves as a seperation between the evaluation at the gauss point of the
 * variables of interest and their use in the assembly, which is carried out
 * by the assembler functions. For more information on this design, the reader
 * can consult deal.II step-9
 * "https://www.dealii.org/current/doxygen/deal.II/step_9.html". In this latter
 * example, the scratch is a struct instead of a templated class because of the
 * simplicity of step-9.
 *
 * @tparam dim An integer that denotes the dimension of the space in which
 * the flow is solved
 *  @ingroup solvers
 **/

template <int dim>
class HeatTransferScratchData
{
public:
  /**
   * @brief Constructor. The constructor creates the fe_values that will be used
   * to fill the member variables of the scratch. It also allocated the
   * necessary memory for all member variables. However, it does not do any
   * evalution, since this needs to be done at the cell level.
   *
   * @param fe The FESystem used to solve the Navier-Stokes equations
   *
   * @param quadrature The quadrature to use for the assembly
   *
   * @param mapping The mapping of the domain in which the Navier-Stokes equations are solved
   *
   */
  HeatTransferScratchData(const FiniteElement<dim> &fe_ht,
                          const Quadrature<dim> &   quadrature,
                          const Mapping<dim> &      mapping,
                          const FiniteElement<dim> &fe_navier_stokes)
    : fe_values_T(mapping,
                  fe_ht,
                  quadrature,
                  update_values | update_quadrature_points | update_JxW_values |
                    update_gradients | update_hessians)
    , fe_values_navier_stokes(mapping,
                              fe_navier_stokes,
                              quadrature,
                              update_values)
  {
    allocate();
  }

  /**
   * @brief Copy Constructor. Same as the main constructor.
   *  This constructor only uses the other scratch to build the FeValues, it
   * does not copy the content of the other scratch into itself since, by
   * definition of the WorkStream mechanism it is assumed that the content of
   * the scratch will be reset on a cell basis.
   *
   * @param fe The FESystem used to solve the Navier-Stokes equations
   *
   * @param quadrature The quadrature to use for the assembly
   *
   * @param mapping The mapping of the domain in which the Navier-Stokes equations are solved
   */
  HeatTransferScratchData(const HeatTransferScratchData<dim> &sd)
    : fe_values_T(sd.fe_values_T.get_mapping(),
                  sd.fe_values_T.get_fe(),
                  sd.fe_values_T.get_quadrature(),
                  update_values | update_quadrature_points | update_JxW_values |
                    update_gradients | update_hessians)
    , fe_values_navier_stokes(sd.fe_values_navier_stokes.get_mapping(),
                              sd.fe_values_navier_stokes.get_fe(),
                              sd.fe_values_navier_stokes.get_quadrature(),
                              update_values)
  {
    allocate();
  }


  /** @brief Allocates the memory for the scratch
   *
   * This function allocates the necessary memory for all members of the scratch
   *
   */
  void
  allocate();

  /** @brief Reinitialize the content of the scratch
   *
   * Using the FeValues and the content ofthe solutions, previous solutions and
   * solutions stages, fills all of the class member of the scratch
   *
   * @param cell The cell over which the assembly is being carried.
   * This cell must be compatible with the fe which is used to fill the FeValues
   *
   * @param current_solution The present value of the solution for [u,p]
   *
   * @param previous_solutions The solutions at the previous time steps
   *
   * @param solution_stages The solution at the intermediary stages (for SDIRK methods)
   *
   * @param source_function The function describing the tracer source term
   *
   */

  template <typename VectorType>
  void
  reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
         const VectorType &                                    current_solution,
         const std::vector<TrilinosWrappers::MPI::Vector> &previous_solutions,
         const std::vector<VectorType> &                   solution_stages,
         Function<dim> *                                   source_function,
         const std::shared_ptr<parallel::DistributedTriangulationBase<dim>>
                                       triangulation,
         MultiphysicsInterface<dim> *  multiphysics,
         TrilinosWrappers::MPI::Vector evaluation_point)
  {
    fe_values_T.reinit(cell);
    auto &fe_T = this->fe_values_T.get_fe();

    if (dim == 2)
      this->cell_size = std::sqrt(4. * cell->measure() / M_PI) / fe_T.degree;
    else if (dim == 3)
      this->cell_size = pow(6 * cell->measure() / M_PI, 1. / 3.) / fe_T.degree;

    fe_values_T.get_function_gradients(current_solution, temperature_gradients);

    const DoFHandler<dim> *dof_handler_fluid =
      multiphysics->get_dof_handler(PhysicsID::fluid_dynamics);

    typename DoFHandler<dim>::active_cell_iterator velocity_cell(
      &(*triangulation), cell->level(), cell->index(), dof_handler_fluid);

    fe_values_navier_stokes.reinit(velocity_cell);

    if (multiphysics->fluid_dynamics_is_block())
      {
        fe_values_navier_stokes[velocities].get_function_values(
          *multiphysics->get_block_solution(PhysicsID::fluid_dynamics),
          velocity_values);
        fe_values_navier_stokes[velocities].get_function_gradients(
          *multiphysics->get_block_solution(PhysicsID::fluid_dynamics),
          velocity_gradient_values);
      }
    else
      {
        fe_values_navier_stokes[velocities].get_function_values(
          *multiphysics->get_solution(PhysicsID::fluid_dynamics),
          velocity_values);
        fe_values_navier_stokes[velocities].get_function_gradients(
          *multiphysics->get_solution(PhysicsID::fluid_dynamics),
          velocity_gradient_values);
      }

    // Gather present value
    fe_values_T.get_function_values(evaluation_point,
                                    present_temperature_values);


    // Gather present laplacian
    fe_values_T.get_function_laplacians(evaluation_point,
                                        present_temperature_laplacians);
    // Gather the previous time steps for heat transfer depending on
    // the number of stages of the time integration method
    for (unsigned int p = 0; p < previous_solutions.size(); ++p)
      {
        fe_values_T.get_function_values(previous_solutions[p],
                                        previous_T_values[p]);
        fe_values_T.get_function_gradients(previous_solutions[p],
                                           previous_T_gradients[p]);


        //     this->fe_values_tracer.get_function_values(previous_solutions[p],
        //                                            previous_tracer_values[p]);
      }



    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        this->JxW[q] = this->fe_values_T.JxW(q);

        for (unsigned int k = 0; k < n_dofs; ++k)
          {
            // Shape function
            this->phi[q][k]           = this->fe_values_T.shape_value(k, q);
            this->grad_phi[q][k]      = this->fe_values_T.shape_grad(k, q);
            this->hess_phi[q][k]      = this->fe_values_T.shape_hessian(k, q);
            this->laplacian_phi[q][k] = trace(this->hess_phi[q][k]);
          }
      }
  }

  template <typename VectorType>
  void
  reinit_velocity(const typename DoFHandler<dim>::active_cell_iterator &cell,
                  const VectorType &current_solution)
  {
    this->fe_values_navier_stokes.reinit(cell);

    this->fe_values_navier_stokes[velocities].get_function_values(
      current_solution, velocity_values);
  }


  // FEValues for the Tracer problem
  FEValues<dim> fe_values_T;
  unsigned int  n_dofs;
  unsigned int  n_q_points;
  double        cell_size;

  // Quadrature
  std::vector<double>     JxW;
  std::vector<Point<dim>> quadrature_points;

  // Shape functions and gradients
  std::vector<double>         phi_T;
  std::vector<Tensor<1, dim>> grad_phi_T;
  std::vector<Tensor<2, dim>> hess_phi_T;
  std::vector<double>         laplacian_phi_T;

  // Source term
  std::vector<double> source;

  // Shape functions
  std::vector<std::vector<double>>         phi;
  std::vector<std::vector<Tensor<2, dim>>> hess_phi;
  std::vector<std::vector<double>>         laplacian_phi;
  std::vector<std::vector<Tensor<1, dim>>> grad_phi;

  std::vector<double> phase_values;


  /**
   * Scratch component for the Navier-Stokes component
   */
  FEValuesExtractors::Vector velocities;
  // This FEValues must mandatorily be instantiated for the velocity
  FEValues<dim>               fe_values_navier_stokes;
  std::vector<Tensor<1, dim>> velocity_values;
  std::vector<Tensor<2, dim>> velocity_gradient_values;

  std::vector<double>                      present_temperature_values;
  std::vector<Tensor<1, dim>>              temperature_gradients;
  std::vector<double>                      present_temperature_laplacians;
  std::vector<double>                      present_face_temperature_values;
  std::vector<std::vector<double>>         previous_T_values;
  std::vector<std::vector<Tensor<1, dim>>> previous_T_gradients;
};

#endif

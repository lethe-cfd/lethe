/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2019 by the Lethe authors
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

#include <core/bdf.h>
#include <core/parameters.h>

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping.h>

#include <deal.II/numerics/vector_tools.h>


#ifndef lethe_navier_stokes_scratch_data_h
#  define lethe_navier_stokes_scratch_data_h

using namespace dealii;

template <int dim>
class NavierStokesScratchData
{
public:
  NavierStokesScratchData(const FESystem<dim> &  fe,
                          const Quadrature<dim> &quadrature,
                          const Mapping<dim> &   mapping)
    : fe_values(mapping,
                fe,
                quadrature,
                update_values | update_quadrature_points | update_JxW_values |
                  update_gradients | update_hessians)
  {
    allocate();
    gather_free_surface = false;
  };

  NavierStokesScratchData(const NavierStokesScratchData<dim> &sd)
    : fe_values(sd.fe_values.get_mapping(),
                sd.fe_values.get_fe(),
                sd.fe_values.get_quadrature(),
                update_values | update_quadrature_points | update_JxW_values |
                  update_gradients | update_hessians)
  {
    allocate();
    if (sd.gather_free_surface)
      enable_free_surface(sd.fe_values_free_surface->get_fe(),
                          sd.fe_values_free_surface->get_quadrature(),
                          sd.fe_values_free_surface->get_mapping());
  };


  void
  allocate();


  template <typename VectorType>
  void
  reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
         const VectorType &                                    current_solution,
         const std::vector<VectorType> &previous_solutions,
         const std::vector<VectorType> &solution_stages,
         Function<dim> *                forcing_function,
         Tensor<1, dim>                 beta_force)
  {
    this->fe_values.reinit(cell);

    quadrature_points = this->fe_values.get_quadrature_points();
    auto &fe          = this->fe_values.get_fe();

    forcing_function->vector_value_list(quadrature_points, this->rhs_force);

    // Establish the force vector
    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (int d = 0; d < dim; ++d)
          {
            const unsigned int component_i =
              fe.system_to_component_index(d).first;
            this->force[q][d] = this->rhs_force[q](component_i);
          }
        // Correct force to include the dynamic forcing term for flow
        // control
        force[q] = force[q] + beta_force;
      }

    if (dim == 2)
      this->cell_size = std::sqrt(4. * cell->measure() / M_PI) / fe.degree;
    else if (dim == 3)
      this->cell_size = pow(6 * cell->measure() / M_PI, 1. / 3.) / fe.degree;

    // Gather velocity (values, gradient and laplacian)
    this->fe_values[velocities].get_function_values(current_solution,
                                                    this->velocity_values);
    this->fe_values[velocities].get_function_gradients(
      current_solution, this->velocity_gradients);
    this->fe_values[velocities].get_function_laplacians(
      current_solution, this->velocity_laplacians);
    for (unsigned int q = 0; q < this->n_q_points; ++q)
      {
        this->velocity_divergences[q] = trace(this->velocity_gradients[q]);
      }

    // Gather previous velocities
    for (unsigned int p = 0; p < previous_solutions.size(); ++p)
      {
        this->fe_values[velocities].get_function_values(
          previous_solutions[p], previous_velocity_values[p]);
      }

    // Gather velocity stages
    for (unsigned int s = 0; s < solution_stages.size(); ++s)
      {
        this->fe_values[velocities].get_function_values(
          solution_stages[s], stages_velocity_values[s]);
      }


    // Gather pressure (values, gradient)
    fe_values[pressure].get_function_values(current_solution,
                                            this->pressure_values);
    fe_values[pressure].get_function_gradients(current_solution,
                                               this->pressure_gradients);


    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        this->JxW[q] = this->fe_values.JxW(q);
        for (unsigned int k = 0; k < n_dofs; ++k)
          {
            // Velocity
            this->phi_u[q][k] = this->fe_values[velocities].value(k, q);
            this->div_phi_u[q][k] =
              this->fe_values[velocities].divergence(k, q);
            this->grad_phi_u[q][k] = this->fe_values[velocities].gradient(k, q);
            this->hess_phi_u[q][k] = this->fe_values[velocities].hessian(k, q);
            for (int d = 0; d < dim; ++d)
              this->laplacian_phi_u[q][k][d] = trace(this->hess_phi_u[q][k][d]);
            // Pressure
            this->phi_p[q][k]      = this->fe_values[pressure].value(k, q);
            this->grad_phi_p[q][k] = this->fe_values[pressure].gradient(k, q);
          }
      }
  }

  void
  enable_free_surface(const FiniteElement<dim> &fe,
                      const Quadrature<dim> &   quadrature,
                      const Mapping<dim> &      mapping)
  {
    gather_free_surface    = true;
    fe_values_free_surface = std::make_shared<FEValues<dim>>(
      mapping, fe, quadrature, update_values | update_gradients);

    // Free surface
    phase_values = std::vector<double>(this->n_q_points);
    previous_phase_values =
      std::vector<std::vector<double>>(maximum_number_of_previous_solutions(),
                                       std::vector<double>(this->n_q_points));
    phase_gradient_values = std::vector<Tensor<1, dim>>(this->n_q_points);
  }


  template <typename VectorType>
  void
  reinit_free_surface(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const VectorType &                                    current_solution,
    const std::vector<VectorType> &                       previous_solutions,
    const std::vector<VectorType> &                       solution_stages)
  {
    // Gather phase fraction (values, gradient)
    this->fe_values_free_surface->get_function_values(current_solution,
                                                      this->phase_values);
    this->fe_values_free_surface->get_function_gradients(
      current_solution, this->phase_gradient_values);

    // Gather previous phase fraction values
    for (unsigned int p = 0; p < previous_solutions.size(); ++p)
      {
        this->fe_values_free_surface->get_function_values(
          previous_solutions[p], previous_phase_values[p]);
      }
  }


  FEValues<dim>              fe_values;
  unsigned int               n_dofs;
  unsigned int               n_q_points;
  double                     cell_size;
  FEValuesExtractors::Vector velocities;
  FEValuesExtractors::Scalar pressure;

  std::vector<Vector<double>> rhs_force;
  Tensor<1, dim>              beta_force;
  std::vector<Tensor<1, dim>> force;

  // Quadrature
  std::vector<double>     JxW;
  std::vector<Point<dim>> quadrature_points;

  // Velocity and pressure values
  std::vector<Tensor<1, dim>>              velocity_values;
  std::vector<double>                      velocity_divergences;
  std::vector<Tensor<2, dim>>              velocity_gradients;
  std::vector<Tensor<1, dim>>              velocity_laplacians;
  std::vector<double>                      pressure_values;
  std::vector<Tensor<1, dim>>              pressure_gradients;
  std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;
  std::vector<std::vector<Tensor<1, dim>>> stages_velocity_values;


  // Shape functions
  std::vector<std::vector<double>>         div_phi_u;
  std::vector<std::vector<Tensor<1, dim>>> phi_u;
  std::vector<std::vector<Tensor<3, dim>>> hess_phi_u;
  std::vector<std::vector<Tensor<1, dim>>> laplacian_phi_u;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
  std::vector<std::vector<double>>         phi_p;
  std::vector<std::vector<Tensor<1, dim>>> grad_phi_p;


  // Phase values for free surface
  bool                             gather_free_surface;
  unsigned int                     n_dofs_free_surface;
  std::vector<double>              phase_values;
  std::vector<std::vector<double>> previous_phase_values;
  std::vector<Tensor<1, dim>>      phase_gradient_values;
  // This is stored as a shared_ptr because it is only instantiated when needed
  std::shared_ptr<FEValues<dim>> fe_values_free_surface;
};

#endif // LETHE_SCRATCH_DATA_H

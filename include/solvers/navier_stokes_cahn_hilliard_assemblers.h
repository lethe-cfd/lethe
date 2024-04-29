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
 * ---------------------------------------------------------------------*/


#include <core/simulation_control.h>

#include <solvers/auxiliary_physics.h>
#include <solvers/copy_data.h>
#include <solvers/navier_stokes_assemblers.h>
#include <solvers/navier_stokes_scratch_data.h>

#ifndef lethe_navier_stokes_cahn_hilliard_assemblers_h
#  define lethe_navier_stokes_cahn_hilliard_assemblers_h


/**
 * @brief Class that assembles the core of the Navier-Stokes equation with
 * free surface using Cahn-Hilliard modeling.
 * According to the following weak form:
 * \f$ \nabla \cdot \mathbf{u} + \rho \mathbf{u} \cdot \nabla
 * \mathbf{u} + \nabla p - \mu \nabla\cdot (\nabla \mathbf{u} +
 * (\nabla \mathbf{u})^T)=0 \f$ with an SUPG and PSPG stabilization
 *
 * @tparam dim An integer that denotes the number of spatial dimensions
 *
 * @ingroup assemblers
 */
template <int dim>
class GLSNavierStokesCahnHilliardAssemblerCore
  : public NavierStokesAssemblerBase<dim>
{
public:
  GLSNavierStokesCahnHilliardAssemblerCore(
    std::shared_ptr<SimulationControl> simulation_control,
    const SimulationParameters<dim>   &nsparam)
    : simulation_control(simulation_control)
    , cahn_hilliard_parameters(nsparam.multiphysics.cahn_hilliard_parameters)
  {}

  /**
   * @brief assemble_matrix Assembles the matrix
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_matrix(NavierStokesScratchData<dim>         &scratch_data,
                  StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  /**
   * @brief assemble_rhs Assembles the rhs
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_rhs(NavierStokesScratchData<dim>         &scratch_data,
               StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  const bool                         SUPG = true;
  std::shared_ptr<SimulationControl> simulation_control;
  const Parameters::CahnHilliard     cahn_hilliard_parameters;
};

/**
 * @brief Class that assembles the transient time arising from BDF time
 * integration for the Navier-Stokes equation with
 * free surface using Cahn-Hilliard modeling. For example, if a BDF1 scheme is
 * chosen, the following is assembled
 * \f$\frac{(\rho \mathbf{u})^{t+\Delta t}-(\rho \mathbf{u})^{t}}{\Delta t} \f$
 *
 * @tparam dim An integer that denotes the number of spatial dimensions
 *
 * @ingroup assemblers
 */
template <int dim>
class GLSNavierStokesCahnHilliardAssemblerBDF
  : public NavierStokesAssemblerBase<dim>
{
public:
  GLSNavierStokesCahnHilliardAssemblerBDF(
    std::shared_ptr<SimulationControl> simulation_control)
    : simulation_control(simulation_control)
  {}

  /**
   * @brief assemble_matrix Assembles the matrix
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_matrix(NavierStokesScratchData<dim>         &scratch_data,
                  StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  /**
   * @brief assemble_rhs Assembles the rhs
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_rhs(NavierStokesScratchData<dim>         &scratch_data,
               StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  std::shared_ptr<SimulationControl> simulation_control;
};

template <int dim>
class GLSNavierStokesCahnHilliardAssemblerNonNewtonianCore
  : public NavierStokesAssemblerBase<dim>
{
public:
  GLSNavierStokesCahnHilliardAssemblerNonNewtonianCore(
    std::shared_ptr<SimulationControl> simulation_control,
    const SimulationParameters<dim>   &nsparam)
    : simulation_control(simulation_control)
    , cahn_hilliard_parameters(nsparam.multiphysics.cahn_hilliard_parameters)
  {}

  /**
   * @brief Calculates an approximation of the gradient of the viscosity
   * @param velocity_gradient The velocity gradient tensor on the quadrature point
     @param velocity_hessians The velocity hessian tensor on the quadrture point
     @param non_newtonian_viscosity The viscosity at which the gradient is calculated
     @param d_gamma_dot Th difference in the shear rate magnitude to approximate the
     viscosity variation with a slight change in the shear_rate magnitude
   */
  inline Tensor<1, dim>
  get_kinematic_viscosity_gradient(
    const Tensor<2, dim> &velocity_gradient,
    const Tensor<3, dim> &velocity_hessians,
    const double          shear_rate_magnitude,
    const double          grad_kinematic_viscosity_shear_rate) const
  {
    // Calculates an approximation of the shear rate magnitude gradient using
    // the derived form, since it does not change with rheological models
    Tensor<1, dim> grad_shear_rate;
    for (unsigned int d = 0; d < dim; ++d)
      {
        if (dim == 2)
          {
            for (unsigned int k = 0; k < dim; ++k)
              {
                grad_shear_rate[d] +=
                  2 * (velocity_gradient[k][k] * velocity_hessians[k][d][k]) /
                  shear_rate_magnitude;
              }
            grad_shear_rate[d] +=
              (velocity_gradient[0][1] + velocity_gradient[1][0]) *
              (velocity_hessians[0][d][1] + velocity_hessians[1][d][0]) /
              shear_rate_magnitude;
          }
        else
          {
            for (unsigned int k = 0; k < dim; ++k)
              {
                grad_shear_rate[d] +=
                  2 * (velocity_gradient[k][k] * velocity_hessians[k][d][k]) /
                    shear_rate_magnitude +
                  (velocity_gradient[(k + 1) % dim][(k + 2) % dim] +
                   velocity_gradient[(k + 2) % dim][(k + 1) % dim]) *
                    (velocity_hessians[(k + 1) % dim][d][(k + 2) % dim] +
                     velocity_hessians[(k + 2) % dim][d][(k + 1) % dim]) /
                    shear_rate_magnitude;
              }
          }
      }
    return grad_shear_rate * grad_kinematic_viscosity_shear_rate;
  };

  /**
   * @brief assemble_matrix Assembles the matrix
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_matrix(NavierStokesScratchData<dim>         &scratch_data,
                  StabilizedMethodsTensorCopyData<dim> &copy_data) override;


  /**
   * @brief assemble_rhs Assembles the rhs
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_rhs(NavierStokesScratchData<dim>         &scratch_data,
               StabilizedMethodsTensorCopyData<dim> &copy_data) override;

  /**
   * Enables SUPG stabilization for the Navier-Stokes formulation.
   * We have not found any scenarios where it is relevant not to use SUPG
   * stabilization yet.
   */
  const bool SUPG = true;

  std::shared_ptr<SimulationControl> simulation_control;
  const Parameters::CahnHilliard     cahn_hilliard_parameters;
};

#endif

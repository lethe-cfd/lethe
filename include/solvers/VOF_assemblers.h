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
 * Implementation of free surface with a Volume of Fluid method.
 * Two fluid formulation. The phase indicator "phase" is equal to 0
 * in one fluid and 1 in the other. The free surface is located
 * where "phase" is equal to 0.5.
 *
 * Author: Jeanne Joachim, Polytechnique Montreal, 2021
 */

#include <core/simulation_control.h>

#include <solvers/copy_data.h>
#include <solvers/VOF_scratch_data.h>


#ifndef vof_assemblers_h
#  define vof_assemblers_h


/**
 * @brief A pure virtual class that serves as an interface for all
 * of the assemblers for the VOF solver
 *
 * @tparam dim An integer that denotes the number of spatial dimensions
 *
 * @ingroup assemblers
 */
template <int dim>
class VOFAssemblerBase
{
public:
  /**
   * @brief assemble_matrix Interface for the call to matrix assembly
   * @param scratch_data Scratch data containing the VOF information.
   * It is important to note that the scratch data has to have been re-inited
   * before calling for matrix assembly.
   * @param copy_data Destination where the local_rhs and loc
   */

  virtual void
  assemble_matrix(VOFScratchData<dim> &   scratch_data,
                  StabilizedMethodsCopyData &copy_data) = 0;


  /**
   * @brief assemble_matrix Interface for the call to rhs
   * @param scratch_data Scratch data containing the free surface information.
   * It is important to note that the scratch data has to have been re-inited
   * before calling for matrix assembly.
   * @param copy_data Destination where the local_rhs and loc
   */

  virtual void
  assemble_rhs(VOFScratchData<dim> &   scratch_data,
               StabilizedMethodsCopyData &copy_data) = 0;
};


/**
 * @brief Class that assembles the core of the VOF solver.
 * This class assembles the weak form of: ***********************\\CORRECT THIS ***************************************
 * $$\mathbf{u} \cdot \nabla T - D \nabla^2 =0 $$ with an SUPG
 * stabilziation
 *
 * @tparam dim An integer that denotes the number of spatial dimensions
 *
 * @ingroup assemblers
 */


template <int dim>
class VOFAssemblerCore : public VOFAssemblerBase<dim>
{
public:
  VOFAssemblerCore(std::shared_ptr<SimulationControl> simulation_control,
                      Parameters::PhysicalProperties     physical_properties)
    : simulation_control(simulation_control)
    , physical_properties(physical_properties)
  {}

  /**
   * @brief assemble_matrix Assembles the matrix
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_matrix(VOFScratchData<dim> &   scratch_data,
                  StabilizedMethodsCopyData &copy_data) override;


  /**
   * @brief assemble_rhs Assembles the rhs
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_rhs(VOFScratchData<dim> &   scratch_data,
               StabilizedMethodsCopyData &copy_data) override;

  const bool DCDD = true;

  std::shared_ptr<SimulationControl> simulation_control;
  Parameters::PhysicalProperties     physical_properties;
};

/**
 * @brief Class that assembles the transient time arising from BDF time
 * integration for the VOF equations. For example, if a BDF1 scheme is
 * chosen, the following is assembled ****************************\\CORRECT THIS ********************************
 * $$\frac{\mathbf{T}^{t+\Delta t}-\mathbf{T}^{t}{\Delta t}
 *
 * @tparam dim An integer that denotes the number of spatial dimensions
 *
 * @ingroup assemblers
 */
template <int dim>
class VOFAssemblerBDF : public VOFAssemblerBase<dim>
{
public:
  VOFAssemblerBDF(std::shared_ptr<SimulationControl> simulation_control)
    : simulation_control(simulation_control)
  {}

  /**
   * @brief assemble_matrix Assembles the matrix
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */

  virtual void
  assemble_matrix(VOFScratchData<dim> &   scratch_data,
                  StabilizedMethodsCopyData &copy_data) override;

  /**
   * @brief assemble_rhs Assembles the rhs
   * @param scratch_data (see base class)
   * @param copy_data (see base class)
   */
  virtual void
  assemble_rhs(VOFScratchData<dim> &   scratch_data,
               StabilizedMethodsCopyData &copy_data) override;

  std::shared_ptr<SimulationControl> simulation_control;
};


#endif

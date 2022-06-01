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
 * Author: Bruno Blais, Polytechnique Montreal, 2019-
 */

/*
 * This file defines the parameters in the parameter namespace
 * that pertain to multiphysics simulations
 */


#ifndef lethe_parameters_multiphysics_h
#define lethe_parameters_multiphysics_h

#include <core/parameters.h>

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

namespace Parameters
{
  /** @brief Class to account for different sharpening type:
   *  - constant: the sharpening threshold is the same throughout the
   * simulation,
   *  - adaptative: the sharpening threshold is determined by binary search, to
   * ensure mass conservation of the monitored phase
   */
  enum class SharpeningType
  {
    constant,
    adaptative
  };

  /**
   * @brief Defines the subparameters for free surface peeling/wetting mechanism.
   * Has to be declared before member creation in VOF structure.
   *
   * Peeling/wetting mechanism (on boundaries explicitely stated in the
   * "subsection boundary conditions VOF" of the .prm) works as such:
   *  - Peeling of the higher density occurs if:
   *    o the pressure is below peeling_pressure_value, and
   *    o the pressure gradient is below the peeling_grad_p
   *  - Wetting of the lower density phase occurs if:
   *    o the pressure is above wetting_pressure_value, and
   *    o the boundary is at a wetting_phase_distance from the
   * interface.
   */
  struct VOF_PeelingWetting
  {
    bool enable;

    // Value used as conditions for the peeling mechanism (fluid detached from
    // a solid boundary). Peeling of the higher density fluid occurs where those
    // conditions are met:
    // - the cell is in the domain of the higher density fluid,
    // - the cell pressure value is below peeling_p_value, and
    // - more than half of the quadrature points in the cell have a pressure
    // gradient below peeling_grad_p.
    double peeling_p_value;
    double peeling_grad_p;

    // Value used as conditions for the wetting mechanism (fluid attached to a
    // solid boundary). Wetting of the lower density fluid occurs where those
    // conditions are met:
    // - the cell is in the domain of the lower density fluid,
    // - the cell pressure value is above wetting_p_value, and
    // - the distance (on the phase value) to the interface is above
    // wetting_phase_distance.
    double wetting_p_value;
    double wetting_phase_distance;

    // artificial diffusivity (diffusion coefficient) (in L^2/s) added to the
    // VOF transport equation. This parameter is zero by default, and can be
    // increased to improve the wetting mechanism. See the documentation for
    // more details.
    double diffusivity;

    // Type of verbosity for the peeling-wetting mechanism
    Parameters::Verbosity verbosity;

    static void
    declare_parameters(ParameterHandler &prm);
    void
    parse_parameters(ParameterHandler &prm);
  };

  /**
   * @brief Defines the subparameters for free surface mass conservation.
   * Has to be declared before member creation in VOF structure.
   */
  struct VOF_MassConservation
  {
    bool skip_mass_conservation_fluid_0;
    bool skip_mass_conservation_fluid_1;
    bool monitoring;
    int  id_fluid_monitored;

    // Conservation tolerance on the fluid monitored,
    // used with adaptative Sharpening
    double tolerance;

    static void
    declare_parameters(ParameterHandler &prm);
    void
    parse_parameters(ParameterHandler &prm);
  };

  /**
   * @brief VOF_InterfaceSharpening - Defines the parameters for
   * interface sharpening in the VOF solver.
   */
  struct VOF_InterfaceSharpening
  {
    // Interface sharpening parameters. The sharpening method and parameters are
    // explained in the dam break VOF example:
    // https://github.com/lethe-cfd/lethe/wiki/Dam-break-VOF
    // sharpening_threshold is the phase fraction threshold for sharpening. It
    // should be chosen in the range of (0,1), but generally it is equal to 0.5
    // interface_sharpness is a parameter which defines the sharpness of the
    // interface. It should be chosen in the range of (1,2] sharpening_frequency
    // (integer) is the frequency at which the interface sharpneing is called.
    // Users may set this variable to 1 to call interface sharpening at every
    // step, but a larger integer value could be chosen.

    bool enable;

    Parameters::SharpeningType type;

    // Parameters for constant sharpening
    double sharpening_threshold;

    // Parameters for adaptative sharpening
    double sharpening_threshold_min;
    double sharpening_threshold_max;
    int    max_iterations;

    // Other sharpening parameters
    double interface_sharpness;
    int    sharpening_frequency;

    // Type of verbosity for the interface sharpening calculation
    Parameters::Verbosity verbosity;

    void
    declare_parameters(ParameterHandler &prm);
    void
    parse_parameters(ParameterHandler &prm);
  };

  /**
   * @brief SurfaceTensionForce - Defines the parameters for
   * the calculation of surface tension force in the VOF solver.
   */
  struct VOF_SurfaceTensionForce
  {
    bool enable;
    // Surface tension coefficient.
    // This will be moved to the property manager in another PR.
    double surface_tension_coef;

    double phase_fraction_gradient_filter_value;
    double curvature_filter_value;

    bool output_vof_auxiliary_fields;

    // Type of verbosity for the surface tension force calculation
    Parameters::Verbosity verbosity;

    void
    declare_parameters(ParameterHandler &prm);
    void
    parse_parameters(ParameterHandler &prm);
  };

  /**
   * @brief VOF - Defines the parameters for free surface simulations
   * using the VOF method.
   * Has to be declared before member creation in Multiphysics structure.
   */
  struct VOF
  {
    Parameters::VOF_MassConservation    conservation;
    Parameters::VOF_InterfaceSharpening sharpening;
    Parameters::VOF_PeelingWetting      peeling_wetting;
    Parameters::VOF_SurfaceTensionForce surface_tension_force;

    void
    declare_parameters(ParameterHandler &prm);
    void
    parse_parameters(ParameterHandler &prm);
  };

  /**
   * @brief Multiphysics - the parameters for multiphysics simulations
   * and handles sub-physics parameters.
   */
  struct Multiphysics
  {
    bool fluid_dynamics;
    bool heat_transfer;
    bool tracer;
    bool VOF;

    // subparameters for heat_transfer
    bool viscous_dissipation;
    bool buoyancy_force;

    Parameters::VOF vof_parameters;

    void
    declare_parameters(ParameterHandler &prm);
    void
    parse_parameters(ParameterHandler &prm);
  };
} // namespace Parameters
#endif

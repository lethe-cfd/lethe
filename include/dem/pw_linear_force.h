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

 *
 * Author: Shahab Golshan, Polytechnique Montreal, 2019
 */

#include <iostream>
#include <math.h>
#include <vector>

#include <deal.II/particles/particle.h>

#include "dem/dem_solver_parameters.h"
#include "dem/physical_info_struct.h"
#include "dem/pw_contact_force.h"
#include "dem/pw_contact_info_struct.h"
#include <dem/dem_properties.h>

using namespace dealii;

#ifndef PWLINEARFORCE_H_
#define PWLINEARFORCE_H_

/**
 * Calculation of the linear particle-wall contact force using the
 * information obtained from the fine search and physical properties of
 * particles and walls
 *
 * @note
 *
 * @author Shahab Golshan, Bruno Blais, Polytechnique Montreal 2019-
 */

template <int dim, int spacedim = dim>
class PWLinearForce : public PWContactForce<dim, spacedim> {
public:
  PWLinearForce() {}

  /**
   * Carries out the calculation of the particle-wall contact force using
   * linear (Hookean) model
   *
   * @param pw_pairs_in_contact Required information for calculation of the
   * particle-wall contact force, these information were obtained in the
   * fine search
   * @param physical_properties Physical properties of particles and walls
   */
  virtual void calculate_pw_contact_force(
      std::vector<std::map<int, pw_contact_info_struct<dim, spacedim>>>
          &pw_pairs_in_contact,
      const physical_info_struct<dim> &physical_properties) override;
};

#endif

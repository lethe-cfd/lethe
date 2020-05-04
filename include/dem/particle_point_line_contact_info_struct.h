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
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/particles/particle_iterator.h>

#ifndef PARTICLEPOINTLINECONTACTINFOSTRUCT_H_
#define PARTICLEPOINTLINECONTACTINFOSTRUCT_H_

/**
 * This struct handles the information related to the calculation of the
 * particle-point and particle-line contact forces
 */

using namespace dealii;

template <int dim> struct particle_point_line_contact_info_struct {
  Particles::ParticleIterator<dim> particle;
  double normal_overlap;
  Tensor<1, dim> normal_vector;
  double normal_relative_velocity;
};

#endif /* PARTICLEPOINTLINECONTACTINFOSTRUCT_H_ */

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
#include <deal.II/particles/particle.h>

#ifndef CONTACTINFOSTRUCT_H_
#  define CONTACTINFOSTRUCT_H_

using namespace dealii;

template <int dim, int spacedim>
struct ContactInfoStruct
{
	double normOverlap;
Point<dim> normVec;
double normRelVel;
Point<dim> tangVec;
double tangRelVel;
double tangOverlap;
Particles::ParticleIterator<dim, spacedim> particleI;
Particles::ParticleIterator<dim, spacedim> particleJ;
};

#endif /* CONTACTINFOSTRUCT_H_ */

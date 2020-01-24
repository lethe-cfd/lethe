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


#include <deal.II/particles/particle_handler.h>

#include <dem/parameters_dem.h>

#ifndef LETHE_DEM_H
#  define LETHE_DEM_H

template <int dim>
class DEMSolver
{
  DEMSolver(ParametersDEM<dim> dem_parameters)
  {}

private:
  ParametersDEM<dim>                        parameters;
  parallel::distributed::Triangulation<dim> triangulation;

  Particles::ParticleHandler<dim, dim> particle_handler;
  Particles::PropertyPool              property_pool;
};

#endif

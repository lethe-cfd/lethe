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

#include <deal.II/fe/mapping_q.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/particles/particle_handler.h>
#include <deal.II/particles/property_pool.h>

#include <dem/dem_solver_parameters.h>

#ifndef LETHE_DEM_H
#  define LETHE_DEM_H

template <int dim>
class DEMSolver
{
public:
  DEMSolver(DEMSolverParameters<dim> dem_parameters);

  void
  solve();

private:
  void read_mesh();

  MPI_Comm           mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout;
  DEMSolverParameters<dim> parameters;

  parallel::distributed::Triangulation<dim> triangulation;

  Particles::PropertyPool              property_pool;
  MappingQGeneric<dim>                 mapping;
  Particles::ParticleHandler<dim, dim> particle_handler;

};

#endif

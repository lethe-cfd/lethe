/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2021 by the Lethe authors
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

#include <dem/dem_properties.h>
#include <dem/dem_solver_parameters.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/particles/particle_handler.h>

#include <iostream>
#include <vector>

using namespace dealii;

#ifndef lagrangian_post_processing_h
#  define lagrangian_post_processing_h

/**
 * This class deals with Lagrangian post-processing. At the moment, calculation
 * of time-averaged particles velocity and granular temperature are added to the
 * class.
 *
 * @todo Add tracer particles
 *
 * @author Shahab Golshan, Polytechnique Montreal 2019-
 */

template <int dim>
class LagrangianPostProcessing
{
public:
  LagrangianPostProcessing<dim>();

  /**
   * Carries out the calculation of the average particles velocity in each local
   * cell. These values are summed up during the post-processing steps (imposed
   * by post-processing frequency), and finally divided by the sampling counter
   * to calculate time-averaged particles velocity distribution.
   *
   * @param triangulation Triangulation
   * @param particle_handler Particle handler
   * @param dem_parameters DEM parameters for setting the names of the output
   * files
   * @param step_number DEM step number
   */
  void
  calculate_average_particles_velocity(
    const parallel::distributed::Triangulation<dim> &triangulation,
    const Particles::ParticleHandler<dim> &          particle_handler,
    const DEMSolverParameters<dim> &                 dem_parameters,
    const unsigned int &                             step_number);

  /**
   * Carries out the calculation of the granular temperature in each local cell.
   * These values are summed up during the post-processing steps (imposed by
   * post-processing frequency), and finally divided by the sampling counter to
   * calculate time-averaged granular temperature distribution.
   *
   * @param triangulation Triangulation
   * @param particle_handler Particle handler
   * @param dem_parameters DEM parameters for setting the names of the output
   * files
   * @param step_number DEM step number
   */
  void
  calculate_average_granular_temperature(
    const parallel::distributed::Triangulation<dim> &triangulation,
    const Particles::ParticleHandler<dim> &          particle_handler,
    const DEMSolverParameters<dim> &                 dem_parameters,
    const unsigned int &                             step_number);

private:
  /**
   * Carries out the calculation of average particles velocity for a single
   * cell. It is defined as a separate private function since the
   * calculate_average_granular_temperature function also needs to obtain the
   * average velocity in the cell.
   *
   * @param cell A single cell in the triangulation
   * @param particle_handler
   * @return A tensor which stores the average particles velocity in the input cell
   */
  Tensor<1, dim>
  calculate_cell_average_particles_velocity(
    const typename parallel::distributed::Triangulation<dim>::cell_iterator
      &                                    cell,
    const Particles::ParticleHandler<dim> &particle_handler);
};

#endif /* lagrangian_post_processing_h */

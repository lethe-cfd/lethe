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

#include <deal.II/distributed/tria.h>

#include <deal.II/particles/particle_handler.h>

#include <dem/dem_solver_parameters.h>
#include <dem/insertion.h>

#ifndef NONUNIFORMINSERTION_H_
#define NONUNIFORMINSERTION_H_

/**
 * Non-uniform insertion of particles in a rectangular box
 *
 * @note
 *
 * @author Shahab Golshan, Bruno Blais, Polytechnique Montreal 2019-
 */

template <int dim> class NonUniformInsertion : public Insertion<dim> {
public:
  /**
   * The constructor to the insertion class calculates the maximum number of
   * particles (maximum_particle_number) which can be inserted in the insertion
   * box and prints a warning if the number of particles to be inserted at this
   * step exceeds this calculated number. Note that in the current version only
   * the insertion in a rectangular box is defined
   *
   * @param dem_parameters DEM parameters declared in the .prm file)
   */
  NonUniformInsertion<dim>(const DEMSolverParameters<dim> &dem_parameters);

  /**
   * Carries out the insertion of particles by discretizing and looping over the
   * insertion box, finding the initial position of particles in the insertion
   * box, their id and other initial properties of the particles. This virtual
   * function that serves as a template for all insertion functions.
   *
   * @param particle_handler The particle handler of particles which are being
   * inserted
   * @param triangulation Triangulation to access the cells in which the
   * particles are inserted
   * @param property_pool Property pool of particles
   * @param dem_parameters DEM parameters declared in the .prm file)
   */
  virtual void insert(Particles::ParticleHandler<dim> &particle_handler,
                      const Triangulation<dim> &triangulation,
                      Particles::PropertyPool &property_pool,
                      const DEMSolverParameters<dim> &dem_parameters) override;

private:
  /**
   * Creates a vector of random numbers with size of particles which are going
   * to be inserted at each insertion step
   *
   * @param inserted_this_step Number of particles which are going to be
   * inserted at this step
   * @param random_number_range The range in which the random numbers will be
   * generated
   * @param random_number_seed random number seed
   * @return A vector of random numbers)
   */
  std::vector<double>
  create_random_number_container(const int &inserted_this_step,
                                 const double &random_number_range,
                                 const int &random_number_seed);
};

#endif /* NONUNIFORMINSERTION_H_ */

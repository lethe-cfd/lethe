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
 * Authors: Carole-Anne Daunais, Valérie Bibeau, Polytechnique Montreal, 2020
 * Jeanne Joachim, Polytechnique Montreal, 2020-
 */

#ifndef lethe_serial_solid_base_h
#define lethe_serial_solid_base_h


// Lethe Includes
#include <core/parameters.h>
#include <core/solid_objects_parameters.h>

// Dealii Includes
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/particles/particle_handler.h>


using namespace dealii;

/**
 * A base class that generates a particle handler for the solid
 *
 * @tparam dim An integer that denotes the dimension of the space in which
 * the flow is solved
 * @tparam spacedim An integer that denotes the dimension of the space occupied
 * by the embedded solid
 *
 * @author Carole-Anne Daunais, Valerie Bibeau, 2020
 */

template <int dim, int spacedim = dim>
class SerialSolid
{
public:
  // Member functions
  SerialSolid(std::shared_ptr<Parameters::SolidObject<spacedim>> &param,unsigned int id);

  /**
   * @brief Manages solid triangulation and particles setup
   */
  void
  initial_setup();

  /**
   * @brief Generates a solid triangulation from a dealii or gmsh mesh
   */
  void
  setup_triangulation(const bool restart);

  /**
   * @brief Set-ups the displacement vector to store the motion of the triangulation
   */
  void
  setup_displacement();

  /**
   * @brief Loads a solid triangulation from a restart file
   */
  void
  load_triangulation(const std::string filename_tria);

  /**
   * @return shared_ptr of the solid triangulation
   */
  std::shared_ptr<Triangulation<dim, spacedim>>
  get_solid_triangulation();

  /**
   * @return the reference to the solid dof handler
   */
  DoFHandler<dim, spacedim> &
  get_solid_dof_handler();

  /**
   * @return the reference to the displacement dof handler
   */
  DoFHandler<dim, spacedim> &
  get_displacement_dof_handler();

  /**
   * @return the reference to the displacement vector
   */
  Vector<double> &
  get_displacement_vector();

  /**
   * @return Function<spacedim> of the solid velocity
   */
  Function<spacedim> *
  get_solid_velocity();

  /**
   * @return id of the solid
   */
  unsigned int
  get_solid_id();

  /**
   * @brief Moves the vertices of the solid triangulation. This function
   * uses an Runge-Kutta 4 explicit time integrator to displace the vertices
   * of the solid triangulation and stores the displacement in an array
   * in order to allow correct checkpointing of the triangulation.
   *               See
   https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
               for more details
   *
   * @param time_step The time_step value for this iteration
   *
   * @param initial_time The initial time (time t) of the timestep. This is used to
   * set the time of the velocity function.
   */
  void
  move_solid_triangulation(double time_step, double initial_time);

  /**
   * @brief Moves the dofs of the solid_triangulation by using the displacement vector.
   * This is only use to move the solid triangulation at the correct location
   * when the simulation is restarted.
   */
  void
  move_solid_triangulation_with_displacement();

  /**
   * @brief Rotates the grid by a given angle around an axis
   */
  void
  rotate_grid(double angle, int axis);

  /**
   * @brief read solid base triangulation checkpoint
   */
  void
  read_checkpoint(std::string prefix_name);

  /**
   * @brief write solid base triangulation checkpoint
   */
  void
  write_checkpoint(std::string prefix_name);


private:
  // Member variables
  MPI_Comm           mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  unsigned int id;

  std::shared_ptr<Triangulation<dim, spacedim>> solid_tria;
  DoFHandler<dim, spacedim>                     solid_dh;

  std::shared_ptr<FiniteElement<dim, spacedim>> fe;

  std::shared_ptr<Mapping<dim, spacedim>> solid_mapping;

  DoFHandler<dim, spacedim>                displacement_dh;
  std::shared_ptr<FESystem<dim, spacedim>> displacement_fe;
  Vector<double>                           displacement;

  std::shared_ptr<Parameters::SolidObject<spacedim>> &param;

  Function<spacedim> *velocity;
};

#endif

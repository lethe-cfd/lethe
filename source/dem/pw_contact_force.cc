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
 * Author: Shahab Golshan, Polytechnique Montreal, 2019
 */

#include <dem/pw_contact_force.h>

// Updates the contact information (contact_info) based on the new information
// of particles pair in the current time step
template <int dim>
void
PWContactForce<dim>::update_contact_information(
  pw_contact_info_struct<dim> &  contact_info,
  const ArrayView<const double> &particle_properties,
  const double &                 dt)
{
  auto               normal_vector = contact_info.normal_vector;
  const unsigned int boundary_id   = contact_info.boundary_id;

  // Using velocity and angular velocity of particle as
  // local vectors
  Tensor<1, dim> particle_velocity;
  particle_velocity[0] = particle_properties[DEM::PropertiesIndex::v_x];
  particle_velocity[1] = particle_properties[DEM::PropertiesIndex::v_y];
  if (dim == 3)
    {
      particle_velocity[2] = particle_properties[DEM::PropertiesIndex::v_z];
    }

  Tensor<1, dim> particle_omega;
  particle_omega[0] = particle_properties[DEM::PropertiesIndex::omega_x];
  particle_omega[1] = particle_properties[DEM::PropertiesIndex::omega_y];
  if (dim == 3)
    {
      particle_omega[2] = particle_properties[DEM::PropertiesIndex::omega_z];
    }

  // Defining relative contact velocity
  Tensor<1, dim> contact_relative_velocity;
  if (dim == 3)
    {
      contact_relative_velocity =
        particle_velocity -
        this->boundary_translational_velocity_map[boundary_id] +
        cross_product_3d((0.5 * particle_properties[DEM::PropertiesIndex::dp] *
                            particle_omega +
                          this->triangulation_radius *
                            this->boundary_rotational_speed_map[boundary_id] *
                            this->boundary_rotational_vector[boundary_id]),
                         normal_vector);
    }
  if (dim == 2)
    {
      contact_relative_velocity =
        particle_velocity - this->triangulation_radius *
                              this->boundary_rotational_speed_map[boundary_id] *
                              cross_product_2d(normal_vector);
    }

  // Calculation of normal relative velocity
  double normal_relative_velocity_value =
    contact_relative_velocity * normal_vector;
  Tensor<1, dim> normal_relative_velocity =
    normal_relative_velocity_value * normal_vector;

  // Calculation of tangential relative velocity
  Tensor<1, dim> tangential_relative_velocity =
    contact_relative_velocity - normal_relative_velocity;

  // Calculation of new tangential_overlap, since this value is
  // history-dependent it needs the value at previous time-step
  // This variable is the main reason that we have iteration over
  // two different vectors (pairs_in_contact and
  // contact_pair_candidates): tangential_overlap of the particles
  // which were already in contact (pairs_in_contact) needs to be
  // modified using its history, while the tangential_overlaps of
  // new particles are equal to zero
  Tensor<1, dim> modified_tangential_overlap =
    contact_info.tangential_overlap + tangential_relative_velocity * dt;

  // Updating the contact_info container based on the new calculated values
  contact_info.normal_relative_velocity     = normal_relative_velocity_value;
  contact_info.tangential_overlap           = modified_tangential_overlap;
  contact_info.tangential_relative_velocity = tangential_relative_velocity;
}

template <int dim>
void
PWContactForce<dim>::get_force_torque()
{
  for (const auto &it : force_on_walls)
    {
      std::cout << "Boundary " << it.first << " :\n"
                << "F = " << it.second << "\nM = " << torque_on_walls[it.first]
                << "\n\n";
    }
}
template <int dim>
std::map<unsigned int, Tensor<1, dim>>
PWContactForce<dim>::initialize(std::map<unsigned int, Tensor<1, dim>> map)
{
  for (const auto &it : map)
    {
      map[it.first] = 0;
    }
  return map;
}


template class PWContactForce<2>;
template class PWContactForce<3>;

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
 */

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/numerics/data_out.h>
 
#include <iostream>
#include <fstream>
#include <cmath>

#include <core/parameters_lagrangian.h>
#include <core/tensors_and_points_dimension_manipulation.h>

#include <dem/force_chains_visualization.h>
#include <dem/particle_particle_contact_force.h>

using namespace DEM;

template <
  int                                                       dim,
  Parameters::Lagrangian::ParticleParticleContactForceModel contact_model,
  Parameters::Lagrangian::RollingResistanceMethod rolling_friction_model>
ParticlesForceChains<dim, contact_model, rolling_friction_model>::
  ParticlesForceChains(const DEMSolverParameters<dim> &dem_parameters)
{
  auto properties  = dem_parameters.lagrangian_physical_properties;
  n_particle_types = properties.particle_type_number;
  effective_youngs_modulus.resize(n_particle_types * n_particle_types);
  effective_shear_modulus.resize(n_particle_types * n_particle_types);
  effective_coefficient_of_restitution.resize(n_particle_types *
                                              n_particle_types);
  effective_coefficient_of_friction.resize(n_particle_types * n_particle_types);
  effective_coefficient_of_rolling_friction.resize(n_particle_types *
                                                   n_particle_types);
  effective_surface_energy.resize(n_particle_types * n_particle_types);
  model_parameter_beta.resize(n_particle_types * n_particle_types);

  for (unsigned int i = 0; i < n_particle_types; ++i)
    {
      const double youngs_modulus_i = properties.youngs_modulus_particle.at(i);
      const double poisson_ratio_i  = properties.poisson_ratio_particle.at(i);
      const double restitution_coefficient_i =
        properties.restitution_coefficient_particle.at(i);
      const double friction_coefficient_i =
        properties.friction_coefficient_particle.at(i);
      const double rolling_friction_coefficient_i =
        properties.rolling_friction_coefficient_particle.at(i);
      const double surface_energy_i = properties.surface_energy_particle.at(i);

      for (unsigned int j = 0; j < n_particle_types; ++j)
        {
          const unsigned int k = i * n_particle_types + j;

          const double youngs_modulus_j =
            properties.youngs_modulus_particle.at(j);
          const double poisson_ratio_j =
            properties.poisson_ratio_particle.at(j);
          const double restitution_coefficient_j =
            properties.restitution_coefficient_particle.at(j);
          const double friction_coefficient_j =
            properties.friction_coefficient_particle.at(j);
          const double rolling_friction_coefficient_j =
            properties.rolling_friction_coefficient_particle.at(j);
          const double surface_energy_j =
            properties.surface_energy_particle.at(j);

          this->effective_youngs_modulus[k] =
            (youngs_modulus_i * youngs_modulus_j) /
            ((youngs_modulus_j * (1.0 - poisson_ratio_i * poisson_ratio_i)) +
             (youngs_modulus_i * (1.0 - poisson_ratio_j * poisson_ratio_j)) +
             DBL_MIN);

          this->effective_shear_modulus[k] =
            (youngs_modulus_i * youngs_modulus_j) /
            (2.0 * ((youngs_modulus_j * (2.0 - poisson_ratio_i) *
                     (1.0 + poisson_ratio_i)) +
                    (youngs_modulus_i * (2.0 - poisson_ratio_j) *
                     (1.0 + poisson_ratio_j))) +
             DBL_MIN);

          this->effective_coefficient_of_restitution[k] =
            harmonic_mean(restitution_coefficient_i, restitution_coefficient_j);

          this->effective_coefficient_of_friction[k] =
            harmonic_mean(friction_coefficient_i, friction_coefficient_j);

          this->effective_coefficient_of_rolling_friction[k] =
            harmonic_mean(rolling_friction_coefficient_i,
                          rolling_friction_coefficient_j);

          this->effective_surface_energy[k] =
            surface_energy_i + surface_energy_j -
            std::pow(std::sqrt(surface_energy_i) - std::sqrt(surface_energy_j),
                     2);

          double restitution_coefficient_particle_log =
            std::log(this->effective_coefficient_of_restitution[k]);

          this->model_parameter_beta[k] =
            restitution_coefficient_particle_log /
            sqrt(restitution_coefficient_particle_log *
                   restitution_coefficient_particle_log +
                 9.8696);
        }
    }
}

template <
  int                                                       dim,
  Parameters::Lagrangian::ParticleParticleContactForceModel contact_model,
  Parameters::Lagrangian::RollingResistanceMethod rolling_friction_model>
void
ParticlesForceChains<dim, contact_model, rolling_friction_model>::
  calculate_force_chains(
    std::vector<Point<3>> &vertices,
    std::vector<double> &force_normal,  
    DEMContactManager<dim>    &container_manager,
    const double               dt,
    std::vector<Tensor<1, 3>> &torque,
    std::vector<Tensor<1, 3>> &force,
    const Tensor<1, dim>       periodic_offset)
{

    ParticleParticleContactForce force_chains_object;


  auto &local_adjacent_particles = container_manager.local_adjacent_particles;
  auto &ghost_adjacent_particles = container_manager.ghost_adjacent_particles;
  auto &local_periodic_adjacent_particles =
    container_manager.local_periodic_adjacent_particles;
  auto &ghost_periodic_adjacent_particles =
    container_manager.ghost_periodic_adjacent_particles;
  auto &ghost_local_periodic_adjacent_particles =
    container_manager.ghost_local_periodic_adjacent_particles;

  // Define local variables which will be used within the contact calculation
  //  Namely: normal and tangential contact forces, tangential and rolling
  //  torques, normal unit vector of the contact and contact relative velocity
  //  in the normal direction
  Tensor<1, 3> normal_unit_vector;
  Tensor<1, 3> normal_force;
  Tensor<1, 3> tangential_force;
  Tensor<1, 3> particle_one_tangential_torque;
  Tensor<1, 3> particle_two_tangential_torque;
  Tensor<1, 3> rolling_resistance_torque;
  double       normal_relative_velocity_value;
  Tensor<1, 3> tangential_relative_velocity;

  // Contact forces calculations of local-local and local-ghost particle
  // pairs are performed in separate loops

  // Looping over local_adjacent_particles values with iterator
  // adjacent_particles_list
  for (auto &&adjacent_particles_list :
       local_adjacent_particles | boost::adaptors::map_values)
    {
      if (!adjacent_particles_list.empty())
        {
          // Gather information about particle 1 and set it up.
          auto first_contact_info = adjacent_particles_list.begin();
          auto particle_one       = first_contact_info->second.particle_one;
          auto particle_one_properties = particle_one->get_properties();

          types::particle_index particle_one_id =
            particle_one->get_local_index();
          Tensor<1, 3> &particle_one_torque = torque[particle_one_id];
          Tensor<1, 3> &particle_one_force  = force[particle_one_id];

          // Fix particle one location for 2d and 3d
          Point<3> particle_one_location = [&] {
            if constexpr (dim == 3)
              {
                return particle_one->get_location();
              }
            if constexpr (dim == 2)
              {
                return (point_nd_to_3d(particle_one->get_location()));
              }
          }();

          for (auto &&contact_info :
               adjacent_particles_list | boost::adaptors::map_values)
            {
              // Getting information (location and properties) of particle 2 in
              // contact with particle 1
              auto particle_two            = contact_info.particle_two;
              auto particle_two_properties = particle_two->get_properties();

              // Get particle 2 location in dimension independent way
              Point<3> particle_two_location = [&] {
                if constexpr (dim == 3)
                  {
                    return particle_two->get_location();
                  }
                if constexpr (dim == 2)
                  {
                    return (point_nd_to_3d(particle_two->get_location()));
                  }
              }();

              // Calculation of normal overlap
              double normal_overlap =
                0.5 * (particle_one_properties[PropertiesIndex::dp] +
                       particle_two_properties[PropertiesIndex::dp]) -
                particle_one_location.distance(particle_two_location);

              if (normal_overlap > 0.0)
                {
                  // This means that the adjacent particles are in contact



                  // Since the normal overlap is already calculated, we update
                  // this element of the container here. The rest of information
                  // are updated using the following function
                  force_chains_object.update_contact_information(
                    contact_info,
                    tangential_relative_velocity,
                    normal_relative_velocity_value,
                    normal_unit_vector,
                    particle_one_properties,
                    particle_two_properties,
                    particle_one_location,
                    particle_two_location,
                    dt);
                  if constexpr (contact_model ==
                                Parameters::Lagrangian::
                                  ParticleParticleContactForceModel::DMT)
                    {
                      force_chains_object.calculate_DMT_contact(
                        contact_info,
                        tangential_relative_velocity,
                        normal_relative_velocity_value,
                        normal_unit_vector,
                        normal_overlap,
                        particle_one_properties,
                        particle_two_properties,
                        normal_force,
                        tangential_force,
                        particle_one_tangential_torque,
                        particle_two_tangential_torque,
                        rolling_resistance_torque);
                    }

                  if constexpr (contact_model ==
                                Parameters::Lagrangian::
                                  ParticleParticleContactForceModel::linear)
                    {
                      force_chains_object.calculate_linear_contact(contact_info,
                                               tangential_relative_velocity,
                                               normal_relative_velocity_value,
                                               normal_unit_vector,
                                               normal_overlap,
                                               particle_one_properties,
                                               particle_two_properties,
                                               normal_force,
                                               tangential_force,
                                               particle_one_tangential_torque,
                                               particle_two_tangential_torque,
                                               rolling_resistance_torque);
                    }

                  if constexpr (contact_model ==
                                Parameters::Lagrangian::
                                  ParticleParticleContactForceModel::hertz)
                    {
                      force_chains_object.calculate_hertz_contact(
                        contact_info,
                        // rst_contact_info->second.particle_  tangential_relative_velocity,
                        normal_relative_velocity_value,
                        normal_unit_vector,
                        normal_overlap,
                        particle_one_properties,
                        particle_two_properties,
                        normal_force,
                        tangential_force,
                        particle_one_tangential_torque,
                        particle_two_tangential_torque,
                        rolling_resistance_torque);
                    }

                  if constexpr (contact_model ==
                                Parameters::Lagrangian::
                                  ParticleParticleContactForceModel::hertz_JKR)
                    {
                      force_chains_object.calculate_hertz_JKR_contact(
                        contact_info,
                        tangential_relative_velocity,
                        normal_relative_velocity_value,
                        normal_unit_vector,
                        normal_overlap,
                        particle_one_properties,
                        particle_two_properties,
                        normal_force,
                        tangential_force,
                        particle_one_tangential_torque,
                        particle_two_tangential_torque,
                        rolling_resistance_torque);
                    }

                  if constexpr (contact_model ==
                                Parameters::Lagrangian::
                                  ParticleParticleContactForceModel::
                                    hertz_mindlin_limit_force)
                    {
                      force_chains_object.calculate_hertz_mindlin_limit_force_contact(
                        contact_info,
                        tangential_relative_velocity,
                        normal_relative_velocity_value,
                        normal_unit_vector,
                        normal_overlap,
                        particle_one_properties,
                        particle_two_properties,
                        normal_force,
                        tangential_force,
                        particle_one_tangential_torque,
                        particle_two_tangential_torque,
                        rolling_resistance_torque);
                    }

                  if constexpr (contact_model ==
                                Parameters::Lagrangian::
                                  ParticleParticleContactForceModel::
                                    hertz_mindlin_limit_overlap)
                    {
                      force_chains_object.calculate_hertz_mindlin_limit_overlap_contact(
                        contact_info,
                        tangential_relative_velocity,
                        normal_relative_velocity_value,
                        normal_unit_vector,
                        normal_overlap,
                        particle_one_properties,
                        particle_two_properties,
                        normal_force,
                        tangential_force,
                        particle_one_tangential_torque,
                        particle_two_tangential_torque,
                        rolling_resistance_torque);
                    }


                  types::particle_index particle_two_id =
                    particle_two->get_local_index();

                  Tensor<1, 3> &particle_two_torque = torque[particle_two_id];
                  Tensor<1, 3> &particle_two_force  = force[particle_two_id];
                  vertices.push_back(particle_one_location);
                  vertices.push_back(particle_two_location);
                  force_normal.push_back(sqrt(normal_force.norme()));
                
                }

              else
                {
                  // if the adjacent pair is not in contact anymore, only the
                  // tangential overlap is set to zero
                  contact_info.tangential_overlap.clear();
                }
            }
        }
    }
}


  void
  multi_general_cell(Triangulation<1, 3>       &tria,
               const std::vector<Point<3>>  &vertices)
  {
    const unsigned int n_cells = vertices.size()/2;
    std::vector<CellData<1>> cells(n_cells, CellData<1>());
    for (unsigned int i =0; i< n_cells; ++i)
    {
      for (unsigned int j =0; j < 2; ++j)
      {
        cells[i].vertices[j] = 2*i +j ;
        cells[i].material_id = 0;
      }
    };
    tria.create_triangulation(vertices, cells, SubCellData());

  }


template <
  int                                                       dim,
  Parameters::Lagrangian::ParticleParticleContactForceModel contact_model,
  Parameters::Lagrangian::RollingResistanceMethod rolling_friction_model>
void
ParticlesForceChains<dim, contact_model, rolling_friction_model>::
  write_force_chains(
    DEMContactManager<dim>    &container_manager,
    const double               dt,
    std::vector<Tensor<1, 3>> &torque,
    std::vector<Tensor<1, 3>> &force,
    const Tensor<1, dim>       periodic_offset)
{
    std::vector<double> force_normal;
    std::vector<Point<3>> vertices;

    calculate_force_chains(
    vertices,
    force_normal,
    container_manager,
    dt,
    torque,
    force,
    periodic_offset);

    Triangulation<1,3> triangulation;
    multi_general_cell(triangulation, vertices);
    DoFHandler<1,3> force_dh(triangulation);
    DataOut<1,3> data_out;  
    data_out.attach_dof_handler(force_dh);

    Vector<float> force_values(triangulation.n_active_cells());
    for (unsigned int i = 0; i < force_values.size(); ++i)
    {
      force_values[i] = force_normal[i];
    }
    data_out.add_data_vector(force_values, "force");

  // const std::string force_name("Forces");
  // std::vector<int> force_values = {1,2};
  // data_out.add_data_vector(force_values, force_name);
  
  std::cout << "printing something " << std::endl;
  data_out.build_patches();
  std::ofstream out("general_cell_data_out.vtu");
  data_out.write_vtu(out);



}


//template class ParticlesForceChains<
//  2,
//  Parameters::Lagrangian::ParticleParticleContactForceModel::hertz,
//  Parameters::Lagrangian::RollingResistanceMethod::no_resistance>;
//
//template class ParticlesForceChains<
//3,
//Parameters::Lagrangian::ParticleParticleContactForceModel::hertz,
//Parameters::Lagrangian::RollingResistanceMethod::no_resistance>;

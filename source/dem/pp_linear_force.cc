#include <dem/pp_linear_force.h>

using namespace dealii;

template <int dim>
void PPLinearForce<dim>::calculate_pp_contact_force(
    const std::map<int, std::map<int, pp_contact_info_struct<dim>>>
        *pairs_in_contact_info,
    const DEMSolverParameters<dim> &dem_parameters) {
  // Defining physical properties as local variable
  const auto physical_properties = dem_parameters.physicalProperties;

  // Looping over pairs_in_contact_info, which means looping over all the active
  // particles with iterator pairs_in_contact_info_iterator
  for (auto pairs_in_contact_info_iterator = pairs_in_contact_info->begin();
       pairs_in_contact_info_iterator != pairs_in_contact_info->end();
       ++pairs_in_contact_info_iterator) {
    // Now an iterator (contact_information_iterator) on each element of the
    // pairs_in_contact_info vector is defined. This iterator iterates over a
    // map which contains the required information for calculation of the
    // contact force for each particle (i.e. each element of the
    // pairs_in_contact_info vector)

    auto pairs_in_contact_content = &pairs_in_contact_info_iterator->second;
    for (auto contact_information_iterator = pairs_in_contact_content->begin();
         contact_information_iterator != pairs_in_contact_content->end();
         ++contact_information_iterator) {
      // Defining the iterator's second value (map value) as a local
      // parameter
      auto contact_information_iterator_second =
          contact_information_iterator->second;

      // Defining total force of the contact, properties of particles one
      // and two as local parameters
      Tensor<1, dim> total_force;
      auto particle_one_properties =
          contact_information_iterator_second.particle_one->get_properties();
      auto particle_two_properties =
          contact_information_iterator_second.particle_two->get_properties();

      // Calculation of effective mass, radius and Young's modulus of the
      // contact
      double effective_mass =
          (particle_one_properties[DEM::PropertiesIndex::mass] *
           particle_two_properties[DEM::PropertiesIndex::mass]) /
          (particle_one_properties[DEM::PropertiesIndex::mass] +
           particle_two_properties[DEM::PropertiesIndex::mass]);
      double effective_radius =
          (particle_one_properties[DEM::PropertiesIndex::dp] *
           particle_two_properties[DEM::PropertiesIndex::dp]) /
          (2.0 * (particle_one_properties[DEM::PropertiesIndex::dp] +
                  particle_two_properties[DEM::PropertiesIndex::dp]));
      double effective_youngs_modulus =
          physical_properties.Youngs_modulus_particle /
          (2.0 * (1.0 - pow(physical_properties.Poisson_ratio_particle, 2.0)));

      // Calculation of normal and tangential spring and dashpot constants
      // using particle properties
      double normal_spring_constant =
          1.2024 * pow((pow(effective_mass, 0.5) *
                        pow(effective_youngs_modulus, 2.0) * effective_radius *
                        abs(contact_information_iterator_second
                                .normal_relative_velocity)),
                       0.4);
      double tangential_spring_constant =
          1.2024 * pow((pow(effective_mass, 0.5) *
                        pow(effective_youngs_modulus, 2.0) * effective_radius *
                        abs(contact_information_iterator_second
                                .tangential_relative_velocity.norm())),
                       0.4);
      double normal_damping_constant =
          (-2.0 * log(physical_properties.restitution_coefficient_particle) *
           sqrt(effective_mass * normal_spring_constant)) /
          (sqrt((pow(log(physical_properties.restitution_coefficient_particle),
                     2.0)) +
                pow(3.1415, 2.0)));
      double tangential_damping_constant = 0;
      if (physical_properties.restitution_coefficient_particle > 0) {
        tangential_damping_constant =
            (-2.0 * log(physical_properties.restitution_coefficient_particle) *
             sqrt(2.0 / 7.0 * effective_mass * tangential_spring_constant)) /
            (sqrt(pow(3.1415, 2.0) +
                  pow(log(physical_properties.restitution_coefficient_particle),
                      2.0)));
      } else {
        tangential_damping_constant =
            2.0 * sqrt(2.0 / 7.0 * effective_mass * tangential_spring_constant);
      }

      // Calculation of normal force using spring and dashpot normal forces
      Tensor<1, dim> spring_normal_force =
          (normal_spring_constant *
           contact_information_iterator_second.normal_overlap) *
          contact_information_iterator_second.normal_vector;
      Tensor<1, dim> dashpot_normal_force =
          (normal_damping_constant *
           contact_information_iterator_second.normal_relative_velocity) *
          contact_information_iterator_second.normal_vector;
      Tensor<1, dim> normal_force;
      normal_force = spring_normal_force + dashpot_normal_force;

      double maximum_tangential_overlap;
      if (tangential_spring_constant > 0) {
        maximum_tangential_overlap =
            physical_properties.friction_coefficient_particle *
            normal_force.norm() / tangential_spring_constant;
      } else {
        maximum_tangential_overlap = 0.0;
      }

      // Check for gross sliding
      if (contact_information_iterator_second.tangential_overlap.norm() >
          maximum_tangential_overlap) {
        // Gross sliding occurs and the tangential overlap and tangnetial
        // force are limited to Coulumb's criterion
        contact_information_iterator_second.tangential_overlap =
            maximum_tangential_overlap *
            (contact_information_iterator_second.tangential_overlap /
             contact_information_iterator_second.tangential_overlap.norm());
      }
      // Calculation of tangential force using spring and dashpot tangential
      // forces
      Tensor<1, dim> spring_tangential_force =
          tangential_spring_constant *
          contact_information_iterator_second.tangential_overlap;
      Tensor<1, dim> dashpot_tangential_force =
          tangential_damping_constant *
          contact_information_iterator_second.tangential_relative_velocity;
      Tensor<1, dim> tangential_force;
      tangential_force =
          -1.0 * spring_tangential_force + dashpot_tangential_force;

      // Calculation of total force
      total_force = normal_force + tangential_force;

      // Updating the body force of particles in the particle handler
      for (int d = 0; d < dim; ++d) {
        particle_one_properties[DEM::PropertiesIndex::force_x + d] =
            particle_one_properties[DEM::PropertiesIndex::force_x + d] -
            total_force[d];
        particle_two_properties[DEM::PropertiesIndex::force_x + d] =
            particle_two_properties[DEM::PropertiesIndex::force_x + d] +
            total_force[d];
      }

      // Calculation of torque
      // Torque caused by tangential force (tangential_torque)
      Tensor<1, dim> tangential_torque_particle_one,
          tangential_torque_particle_two;

      if (dim == 3) {
        tangential_torque_particle_one = cross_product_3d(
            ((particle_one_properties[DEM::PropertiesIndex::dp] / 2.0) *
             contact_information_iterator_second.normal_vector),
            tangential_force);
        tangential_torque_particle_two = cross_product_3d(
            ((particle_two_properties[DEM::PropertiesIndex::dp] / 2.0) *
             contact_information_iterator_second.normal_vector),
            tangential_force);
      }

      // Rolling resistance torque
      // For calculation of rolling resistance torque, we need to obtain
      // omega_ij using rotational velocities of particles one and two
      Tensor<1, dim> particle_one_angular_velocity,
          particle_two_angular_velocity, omega_ij, omega_ij_direction;
      for (int d = 0; d < dim; ++d) {
        particle_one_angular_velocity[d] =
            particle_one_properties[DEM::PropertiesIndex::omega_x + d];
        particle_two_angular_velocity[d] =
            particle_two_properties[DEM::PropertiesIndex::omega_x + d];
        omega_ij_direction[d] = 0.0;
      }
      omega_ij = particle_one_angular_velocity - particle_two_angular_velocity;
      double omega_ij_value = omega_ij.norm();

      if (omega_ij_value != 0) {
        omega_ij_direction = omega_ij / omega_ij_value;
      }

      // Calculation of rolling resistance torque
      Tensor<1, dim> rolling_resistance_torque =
          -1.0 * physical_properties.rolling_friction_particle *
          effective_radius * normal_force.norm() * omega_ij_direction;

      // Updating the torque acting on particles
      for (int d = 0; d < dim; ++d) {
        particle_one_properties[DEM::PropertiesIndex::M_x + d] =
            particle_one_properties[DEM::PropertiesIndex::M_x + d] -
            tangential_torque_particle_one[d] + rolling_resistance_torque[d];
        particle_two_properties[DEM::PropertiesIndex::M_x + d] =
            particle_two_properties[DEM::PropertiesIndex::M_x + d] -
            tangential_torque_particle_two[d] - rolling_resistance_torque[d];
      }
    }
  }
}

template class PPLinearForce<2>;
template class PPLinearForce<3>;

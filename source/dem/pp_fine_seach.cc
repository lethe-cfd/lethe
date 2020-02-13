#include "dem/pp_fine_search.h"

using namespace dealii;

template <int dim, int spacedim> PPFineSearch<dim, spacedim>::PPFineSearch() {}

template <int dim, int spacedim>
void PPFineSearch<dim, spacedim>::pp_Fine_Search(
    std::vector<std::pair<Particles::ParticleIterator<dim, spacedim>,
                          Particles::ParticleIterator<dim, spacedim>>>
        &contact_pair_candidates,
    std::vector<std::map<int, Particles::ParticleIterator<dim, spacedim>>>
        &pairs_in_contact,
    std::vector<std::map<int, pp_contact_info_struct<dim, spacedim>>>
        &pairs_in_contact_info,
    double dt, Particles::ParticleHandler<dim, spacedim> &particle_handler) {

  // Iterating over all the particles in particle_handler, which shows the first
  // particle in the contact pair

  // ??????????????????????????////
  // what about iterating over pairs_in_contact directly???
  for (auto particle_one = particle_handler.begin();
       particle_one != particle_handler.end(); ++particle_one) {

    // Defining element (particle->get_id())th of the pairs_in_contact vector as
    // a local map (contact_partners_map)
    auto contact_partners_map = pairs_in_contact[particle_one->get_id()];

    // Iterating over each element of pairs_in_contact which contains the second
    // particle in the contact pair
    for (auto map_iterator = contact_partners_map.begin();
         map_iterator != contact_partners_map.end(); map_iterator++) {
      auto particle_two = map_iterator->second;

      // Finding the locations of the particles in contact
      Point<dim, double> particle_one_location = particle_one->get_location();
      Point<dim, double> particle_two_location = particle_two->get_location();

      // Get the total array view to the particle properties once to improve
      // efficiency
      auto particle_one_properties = particle_one->get_properties();
      auto particle_two_properties = particle_two->get_properties();

      // Finding the distance of particles based on their new position, if the
      // particles are still in contact, the distance will be equal to the
      // normal overlap
      double distance = ((particle_one_properties[DEM::PropertiesIndex::dp] +
                          particle_two_properties[DEM::PropertiesIndex::dp]) /
                         2) -
                        particle_one_location.distance(particle_two_location);

      // If the pair is still in contact
      if (distance > 0) {

        // contact_vector shows a vector from location of particle_one to
        // location of particle_two
        Tensor<1, dim> contact_vector =
            (particle_two_location - particle_one_location);

        // Using contact_vector, the contact normal vector is obtained
        Tensor<1, dim> normal_vector = contact_vector / contact_vector.norm();

        // Using velocities and angular velocities of particles one and two as
        // vectors
        Tensor<1, dim> particle_one_velocity{
            {particle_one_properties[DEM::PropertiesIndex::v_x],
             particle_one_properties[DEM::PropertiesIndex::v_y],
             particle_one_properties[DEM::PropertiesIndex::v_z]}};

        Tensor<1, dim> particle_two_velocity{
            {particle_two_properties[DEM::PropertiesIndex::v_x],
             particle_two_properties[DEM::PropertiesIndex::v_y],
             particle_two_properties[DEM::PropertiesIndex::v_z]}};

        Tensor<1, dim> particle_one_omega{
            {particle_one_properties[DEM::PropertiesIndex::omega_x],
             particle_one_properties[DEM::PropertiesIndex::omega_y],
             particle_one_properties[DEM::PropertiesIndex::omega_z]}};

        Tensor<1, dim> particle_two_omega{
            {particle_two_properties[DEM::PropertiesIndex::omega_x],
             particle_two_properties[DEM::PropertiesIndex::omega_y],
             particle_two_properties[DEM::PropertiesIndex::omega_z]}};

        // Defining relative contact velocity
        Tensor<1, dim> contact_relative_velocity;
        if (dim == 3) {
          contact_relative_velocity =
              (particle_one_velocity - particle_two_velocity) +
              (cross_product_3d(
                  (((particle_one_properties[DEM::PropertiesIndex::dp] / 2.0) *
                    particle_one_omega) +
                   ((particle_two_properties[DEM::PropertiesIndex::dp] / 2.0) *
                    particle_two_omega)),
                  normal_vector));
        }
        /*
        if (dim == 2) {
          contact_relative_velocity =
              (particle_one_velocity - particle_two_velocity) +
              (cross_product_2d(
                  (((particle_one_properties[DEM::PropertiesIndex::dp] / 2.0) *
                    particle_one_omega) +
                   ((particle_two_properties[DEM::PropertiesIndex::dp] / 2.0) *
                    particle_two_omega)),
                  normal_vector));
        }
        */

        // Calculation of normal relative velocity. Note that in the following
        // line the product acts as inner product since both sides are vectors,
        // while in the second line the product is scalar and vector product
        double normal_relative_velocity_value =
            contact_relative_velocity * normal_vector;
        Tensor<1, dim> normal_relative_velocity =
            normal_relative_velocity_value * normal_vector;

        // Calculation of tangential relative velocity
        Tensor<1, dim> tangential_relative_velocity =
            contact_relative_velocity - normal_relative_velocity;

        // Calculation of tangential vector using tangential relative velocity
        Tensor<1, dim> tangential_vector{{0, 0, 0}};

        double tangential_relative_velocity_value =
            tangential_relative_velocity.norm();
        if (tangential_relative_velocity_value != 0) {
          tangential_vector =
              tangential_relative_velocity / tangential_relative_velocity_value;
        }

        // Calculation of new tangential_overlap, since this value is
        // history-dependent it needs the value at previous time-step This
        // variable is the main reason that we have iteration over two different
        // vectors (pairs_in_contact and contact_pair_candidates):
        // tangential_overlap of the particles which were already in contact
        // (pairs_in_contact) needs to modified using its history, while the
        // tangential_overlaps of new particles are equal to zero
        double tangential_overlap =
            pairs_in_contact_info[particle_one->get_id()]
                                 [particle_two->get_id()]
                                     .tangential_overlap +
            (tangential_relative_velocity_value * dt);

        // Creating a sample from the pp_contact_info_struct and adding contact
        // info to the sample
        pp_contact_info_struct<dim, spacedim> contact_info;

        contact_info.normal_overlap = distance;
        contact_info.normal_vector = normal_vector;
        contact_info.normal_relative_velocity = normal_relative_velocity_value;
        contact_info.tangential_vector = tangential_vector;
        contact_info.tangential_relative_velocity =
            tangential_relative_velocity_value;
        contact_info.tangential_overlap = tangential_overlap;
        contact_info.particle_one = particle_one;
        contact_info.particle_two = particle_two;

        pairs_in_contact_info[particle_one->get_id()].insert_or_assign(
            particle_two->get_id(), contact_info);
      }

      // If the particles are not in contact anymore (i.e. the contact is
      // finished and distance <= 0), this element should be erased from
      // pairs_in_contact and pairs_in_contact_info:
      else {

        (pairs_in_contact[particle_one->get_id()])
            .erase(particle_two->get_id());
        (pairs_in_contact_info[particle_one->get_id()])
            .erase(particle_two->get_id());
      }
    }
  }

  // Now iterating over contact candidates from broad search. If a pair is in
  // contact (distance > 0) and does not exist in the pairs_in_contact, it is
  // added to the pairs_in_contact
  for (auto contact_pair_candidates_iterator = contact_pair_candidates.begin();
       contact_pair_candidates_iterator != contact_pair_candidates.end();
       ++contact_pair_candidates_iterator) {

    // Get particles one and two from the vector and the total array view to the
    // particle properties once to improve efficiency
    auto particle_one = contact_pair_candidates_iterator->first;
    auto particle_two = contact_pair_candidates_iterator->second;
    auto particle_one_properties = particle_one->get_properties();
    auto particle_two_properties = particle_two->get_properties();

    // Obtaining locations of particles one and two:
    Point<dim, double> particle_one_location = particle_one->get_location();
    Point<dim, double> particle_two_location = particle_two->get_location();

    // Calculation of the distance between particles one and two:
    double distance = ((particle_one_properties[DEM::PropertiesIndex::dp] +
                        particle_two_properties[DEM::PropertiesIndex::dp]) /
                       2) -
                      particle_one_location.distance(particle_two_location);

    // Check to see if particle pair is in contact:
    if (distance > 0) {

      // Check to see if the pair already exists in pairs_in_contact vector or
      // not. Note that the pair shoule be searched in the
      // (particle_one_properties[DEM::PropertiesIndex::id])th element of the
      // vector as well as (particle_two_properties[DEM::PropertiesIndex::id])th
      // element of the vector. If the pair does not exist in these two
      // locations, it should be added to pairs_in_contact
      if (pairs_in_contact[particle_one_properties[DEM::PropertiesIndex::id]]
                  .count(particle_two_properties[DEM::PropertiesIndex::id]) <=
              0 &&
          pairs_in_contact[particle_two_properties[DEM::PropertiesIndex::id]]
                  .count(particle_one_properties[DEM::PropertiesIndex::id]) <=
              0) {
        // Adding the pair in the
        // (particle_one_properties[DEM::PropertiesIndex::id])th element of
        // pairs_in_contact vector with the key of second's particle id
        pairs_in_contact[particle_one_properties[DEM::PropertiesIndex::id]]
            .insert({particle_two_properties[DEM::PropertiesIndex::id],
                     particle_two});

        // contact_vector shows a vector from location of particle_one to
        // location of particle_two
        Tensor<1, dim> contact_vector =
            (particle_two_location - particle_one_location);

        // Using contact_vector, the contact normal vector is obtained
        Tensor<1, dim> normal_vector = contact_vector / contact_vector.norm();

        // Using velocities and angular velocities of particles one and two as
        // vectors
        Tensor<1, dim> particle_one_velocity{
            {particle_one_properties[DEM::PropertiesIndex::v_x],
             particle_one_properties[DEM::PropertiesIndex::v_y],
             particle_one_properties[DEM::PropertiesIndex::v_z]}};

        Tensor<1, dim> particle_two_velocity{
            {particle_two_properties[DEM::PropertiesIndex::v_x],
             particle_two_properties[DEM::PropertiesIndex::v_y],
             particle_two_properties[DEM::PropertiesIndex::v_z]}};

        Tensor<1, dim> particle_one_omega{
            {particle_one_properties[DEM::PropertiesIndex::omega_x],
             particle_one_properties[DEM::PropertiesIndex::omega_y],
             particle_one_properties[DEM::PropertiesIndex::omega_z]}};

        Tensor<1, dim> particle_two_omega{
            {particle_two_properties[DEM::PropertiesIndex::omega_x],
             particle_two_properties[DEM::PropertiesIndex::omega_y],
             particle_two_properties[DEM::PropertiesIndex::omega_z]}};

        // Defining relative contact velocity
        Tensor<1, dim> contact_relative_velocity;
        if (dim == 3) {
          contact_relative_velocity =
              (particle_one_velocity - particle_two_velocity) +
              (cross_product_3d(
                  (((particle_one_properties[DEM::PropertiesIndex::dp] / 2.0) *
                    particle_one_omega) +
                   ((particle_two_properties[DEM::PropertiesIndex::dp] / 2.0) *
                    particle_two_omega)),
                  normal_vector));
        }

        /*
        if (dim == 2) {
          contact_relative_velocity =
              (particle_one_velocity - particle_two_velocity) +
              (cross_product_2d(
                  (((particle_one_properties[DEM::PropertiesIndex::dp] / 2.0) *
                    particle_one_omega) +
                   ((particle_two_properties[DEM::PropertiesIndex::dp] / 2.0) *
                    particle_two_omega)),
                  normal_vector));
        }
        */

        // Calculation of normal relative velocity. Note that in the following
        // line the product acts as inner product since both sides are vectors,
        // while in the second line the product is scalar and vector product
        double normal_relative_velocity_value =
            contact_relative_velocity * normal_vector;
        Tensor<1, dim> normal_relative_velocity =
            normal_relative_velocity_value * normal_vector;

        // Calculation of tangential relative velocity
        Tensor<1, dim> tangential_relative_velocity =
            contact_relative_velocity - normal_relative_velocity;

        // Calculation of tangential vector using tangential relative velocity
        Tensor<1, dim> tangential_vector{{0, 0, 0}};

        double tangential_relative_velocity_value =
            tangential_relative_velocity.norm();
        if (tangential_relative_velocity_value != 0) {
          tangential_vector =
              tangential_relative_velocity / tangential_relative_velocity_value;
        }

        // For new pairs added to pairs_in_contact, the tangential overlap is
        // equal to zero
        double tangential_overlap = 0;

        // Creating a sample from the contact_info_struct and adding contact
        // info to the sample
        pp_contact_info_struct<dim, spacedim> contact_info;

        contact_info.normal_overlap = distance;
        contact_info.normal_vector = normal_vector;
        contact_info.normal_relative_velocity = normal_relative_velocity_value;
        contact_info.tangential_vector = tangential_vector;
        contact_info.tangential_relative_velocity =
            tangential_relative_velocity_value;
        contact_info.tangential_overlap = tangential_overlap;
        contact_info.particle_one = particle_one;
        contact_info.particle_two = particle_two;
        pairs_in_contact_info[particle_one_properties[DEM::PropertiesIndex::id]]
            .insert({particle_two_properties[DEM::PropertiesIndex::id],
                     contact_info});
      }
    }
  }
}

template class PPFineSearch<3, 3>;

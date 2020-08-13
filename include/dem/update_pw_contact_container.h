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

using namespace dealii;

#ifndef UPDATEPWCONTACTCONTAINER_H_
#define UPDATEPWCONTACTCONTAINER_H_

/**
 * Updates the iterators to particles in pw_contact_container (output of pw
 * fine search)
 *
 * @param pw_pairs_in_contact Output of particle-wall fine search
 * @param particle_container Output of update_particle_container function
 */

template <int dim>
void
update_pw_contact_container_iterators(
  std::map<int, std::map<int, pw_contact_info_struct<dim>>>
    &                                                    pw_pairs_in_contact,
  const std::map<int, Particles::ParticleIterator<dim>> &particle_container)
{
  for (auto pw_pairs_in_contact_iterator = pw_pairs_in_contact.begin();
       pw_pairs_in_contact_iterator != pw_pairs_in_contact.end();
       ++pw_pairs_in_contact_iterator)
    {
      int particle_id = pw_pairs_in_contact_iterator->first;

      auto pairs_in_contant_content = &pw_pairs_in_contact_iterator->second;

      for (auto pw_map_iterator = pairs_in_contant_content->begin();
           pw_map_iterator != pairs_in_contant_content->end();
           ++pw_map_iterator)
        {
          pw_map_iterator->second.particle = particle_container.at(particle_id);
        }
    }
}

#endif /* UPDATEPWCONTACTCONTAINER_H_ */

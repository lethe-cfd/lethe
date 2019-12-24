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

#include "dem/parameters_dem.h"
#include "dem/integration.h"
#include "dem/particle_insertion.h"


#ifndef DEMITERATOR_H_
#  define DEMITERATOR_H_

class DEM_iterator
{
public:
  DEM_iterator();
  void
  engine(
    int &,
    dealii::Particles::ParticleHandler<3, 3> &,
    const dealii::Triangulation<3, 3> &,
    int &,
    float &,
    ParametersDEM<3>,
    std::pair<std::vector<std::set<Triangulation<3>::active_cell_iterator>>,
              std::vector<Triangulation<3>::active_cell_iterator>>,
    std::vector<std::tuple<std::pair<Particles::ParticleIterator<3, 3>,
                                     Particles::ParticleIterator<3, 3>>,
                           double,
                           Point<3>,
                           double,
                           Point<3>,
                           double,
                           double>> &,
    std::vector<std::tuple<int,
                           Triangulation<3>::active_cell_iterator,
                           int,
                           Point<3>,
                           Point<3>>>,
    std::vector<std::tuple<std::pair<Particles::ParticleIterator<3, 3>, int>,
                           Point<3>,
                           Point<3>,
                           double,
                           double,
                           double,
                           Point<3>,
                           double>> &, std::vector<std::tuple<std::string, int>>, dealii::Particles::PropertyPool &);

private:
  void forceReinit(Particles::ParticleHandler<3, 3> &);
  // void checkSimBound(Particles::ParticleHandler<3, 3> &, ReadInputScript);
};

#endif /* DEMITERATOR_H_ */

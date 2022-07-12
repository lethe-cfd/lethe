/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 - 2022 by the Lethe authors
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

#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/particles/particle_handler.h>

#include <boost/range/adaptor/map.hpp>

#include <iostream>
#include <vector>

#ifndef data_containers_h
#  define data_containers_h

/**
 * This namespace defines all the data containers and required operations on
 * these containers for the DEM solver. At the moment, I only added overloaded
 * operators here, but finally all the data containers of the DEM solver will be
 * moved here
 */

namespace dem_data_containers
{
  /**
   * Operator overloading to enable using triangulation cells as map keys.
   */

  template <int dim>
  struct cell_comparison
  {
    bool
    operator()(
      const typename dealii::Triangulation<dim>::active_cell_iterator &cell_1,
      const typename dealii::Triangulation<dim>::active_cell_iterator &cell_2)
      const
    {
      return cell_1->id().to_string() < cell_2->id().to_string();
    }
  };

} // namespace dem_data_containers
#endif

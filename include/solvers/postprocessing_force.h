/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 -  by the Lethe authors
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
 * Author: Bruno Blais, Polytechnique Montreal, 2020 -
 */

#ifndef LETHE_FORCECALCULATION_H
#define LETHE_FORCECALCULATION_H

// Base
#include <deal.II/base/quadrature_lib.h>

// Lac
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>

// Dofs
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

// Fe
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

// Lethe includes
#include <core/parameters.h>
#include <solvers/boundary_conditions.h>


using namespace dealii;

// This is a primitive first implementation that could be greatly improved by
// doing a single pass instead of N boundary passes
template <int dim, typename VectorType>
std::vector<Tensor<1,dim> >
calculate_forces(
    const FESystem<dim>                                 &fe,
    const DoFHandler<dim>                               &dof_handler,
    const VectorType                                    &evaluation_point,
    const Parameters::PhysicalProperties                &physical_properties,
    const Parameters::FEM                               &fem_parameters,
    const BoundaryConditions::NSBoundaryConditions<dim> &boundary_conditions,
    const MPI_Comm                                      &mpi_communicator
    );

#endif

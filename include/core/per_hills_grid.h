/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 -  by the Lethe authors
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
 * Author: Audrey Collard-Daigneault, Polytechnique Montreal, 2020 -
 */

#ifndef lethe_per_hills_grid_h
#define lethe_per_hills_grid_h

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/utilities.h>
#include <sstream>

using namespace dealii;

/**
 * @brief PeriodicHillsGrid.The PeriodicHillsGrid class creates an hyper_rectangle and transforms it to
 * obtain the hill geometry with the hill_geometry function.
 * It also attaches a manifold to the geometry.
 */

template <int dim, int spacedim>
class PeriodicHillsGrid
{
public:
  /**
   * @brief Constructor for the PeriodicHillsGrid. At the present moment, the periodic hill
   * cannot be controlled from the parameter file. The Grid is generated as-is.
   */
  PeriodicHillsGrid(const std::string &grid_arguments);

  /**
   * @brief The hill_geometry function calculates all the domain of the geometry with 6
   * polynomials depending the x position. (See Hill Geometry Definition file :
   * https://turbmodels.larc.nasa.gov/Other_LES_Data/2dhill_periodic.html)
   * This code has nondimensionalized geometry, but the coefficients provided
   * need a hill height of 28.
   * This function also does a gradual shifting of the horizontal lines
   * prior to have smaller element on the bottom of the geometry where results
   * are more important.
   *
   * @param p A point in space which will be adapted to the periodic hill geometry
   *
   * @param param Non-linear solver parameters
   *
   */
  Point<spacedim>
  static hill_geometry(const Point<spacedim> &p,
                       double alpha, double spacing_y);

  /**
   * @brief make_grid. The make_grid function generates a hyper rectangle of the size of the domain
   * and then transforms it to the hill geometry. It also constructs the
   * geometry manifold with FunctionManifold and finally sets the manifold.
   *
   * @param triangulation. The triangulation object on which the grid is generated
   */
  void
  make_grid(Triangulation<dim, spacedim> &triangulation);

private:
  std::string grid_arguments;
  double alpha;
  double spacing_y;
  int repetitions_x;
  int repetitions_y;
  int repetitions_z;
};

/**
 * @brief The push_forward & the pull_back classes create the vector_value functions
 * gradient function and because it inherits from Function<spacedim>. (formula
 * is currently Euler and can be changed. See AutoDerivativeFunction
 * documentation)
 */
template <int dim, int spacedim>
class periodic_hills_push_forward : public AutoDerivativeFunction<spacedim>
{
public:
  periodic_hills_push_forward(double alpha, double spacing_y)
    : AutoDerivativeFunction<spacedim>(1e-6, spacedim), alpha(alpha), spacing_y(spacing_y)
  {}

  /**
   * @brief vector_value. This function is used to construct the geometry manifold.
   * It changes the original point (op) of the transformed hyper_rectangle
   * with hill_geometry to a new point (np) of the hill grid with per_hills_grid
   * function.
   *
   * @param p. A point in space
   *
   * @param values. The vector of values which will be calculated at the position p.
   */

  virtual void
  vector_value(const Point<spacedim> &p, Vector<double> &values) const override;

  /**
   * @brief value. The value function does the same thing than vector_value for one component.
   * This implementation is needed to use the gradient function inherited by
   * AutoDerivativeFunction.
   *
   * @param p. An original point in space
   *
   * @param component. The component of the point (x=0, y=1, z=2)
   *
   */
  virtual double
  value(const Point<spacedim> &p, const unsigned int component) const override;

private:
  double alpha;
  double spacing_y;
};


template <int dim, int spacedim>
class periodic_hills_pull_back : public AutoDerivativeFunction<spacedim>
{
public:
  periodic_hills_pull_back(double alpha, double spacing_y)
    : AutoDerivativeFunction<spacedim>(1e-6, spacedim), alpha(alpha), spacing_y(spacing_y)
  {}

  /**
   * \brief vector_value. This vector_value function is used to construct the
   * geometry manifold. It changes the new point (np) of the hill grid to the
   * original point (op) of the transformed hyper_rectangle grid. This function
   * is mandatory to use FunctionManifold. First, it finds the minimum value of
   * y depending the x position and then calculates the op with the inverse of
   * the transformation done by hill_geometry function.
   *
   * \param p. A point in space.
   *
   * @param values. The vector of values which will be calculated at the position p.
   */

  virtual void
  vector_value(const Point<spacedim> &np,
               Vector<double> &       values) const override;

  /**
   * @brief value. The value function does the same thing than vector_value for one component.
   * This implementation is needed to use the gradient function inherited by
   * AutoDerivativeFunction.
   *
   * @param p. An original point in space
   *
   * @param component. The component of the point (x=0, y=1, z=2)
   *
   */
  virtual double
  value(const Point<spacedim> &np, const unsigned int component) const override;

private:
  double alpha;
  double spacing_y;
};

template <int dim, int spacedim>
PeriodicHillsGrid<dim, spacedim>::PeriodicHillsGrid(const std::string &grid_arguments)
{
  this->grid_arguments = grid_arguments;

  // Separate arguments of the string
  std::vector<std::string> arguments;
  std::stringstream s_stream(grid_arguments);
  while(s_stream.good())
  {
    std::string substr;
    getline(s_stream, substr, ';');
    arguments.push_back(substr);
  }

  std::vector<double> arguments_double = dealii::Utilities::string_to_double(arguments);
  alpha = arguments_double[0];
  spacing_y = arguments_double[1];
  repetitions_x = arguments_double[2];
  repetitions_y = arguments_double[3];
  if (dim == 3)
    repetitions_z = arguments_double[4];
}


template <int dim, int spacedim>
void
periodic_hills_push_forward<dim, spacedim>::vector_value(
  const Point<spacedim> &op,
  Vector<double> &       values) const
{
  const Point<spacedim> np =
    PeriodicHillsGrid<dim, spacedim>::hill_geometry(op, alpha, spacing_y);

  std::cout << "Push foward x : " << op[0] << " to " << np[0] << std::endl;

  values(0) = np[0];
  values(1) = np[1];

  if (spacedim == 3)
    values(2) = np[2];
}


template <int dim, int spacedim>
double
periodic_hills_push_forward<dim, spacedim>::value(
  const Point<spacedim> &op,
  const unsigned int     component) const
{
  const Point<spacedim> np =
    PeriodicHillsGrid<dim, spacedim>::hill_geometry(op, alpha, spacing_y);
  return np[component];
}


template <int dim, int spacedim>
void
periodic_hills_pull_back<dim, spacedim>::vector_value(
  const Point<spacedim> &np,
  Vector<double> &       values) const
{
  double       x = np[0];

  const double max_y = 3.035;
  double       min_y;
  double flat_region_length = 5.142;
  double left_hill = 1.929;
  double right_hill = 7.071;


  if (x < left_hill * alpha)
    x = (x / alpha);
  else if (x > alpha * left_hill + flat_region_length)
    x = (x - flat_region_length - alpha * left_hill) / alpha + right_hill;
  else
    x = x - (alpha * left_hill) + left_hill;

  std::cout << "Pull back x : " << np[0] << " to " << x << std::endl;

  if (spacedim == 2)
  {
    min_y = PeriodicHillsGrid<dim, spacedim>::hill_geometry(
      Point<spacedim>(x, 0), alpha, spacing_y)[1];
  }
  else if (spacedim == 3)
  {
    min_y = PeriodicHillsGrid<dim, spacedim>::hill_geometry(
      Point<spacedim>(x, 0, np[2]), alpha, spacing_y)[1];
    values(2) = np[2];
  }

  double y = (np[1] - min_y) / (1 - min_y / max_y);

  if (y <= (max_y/2))
    y = (-(1 - 0.5 * spacing_y) + std::sqrt(std::pow((1 - 0.5 * spacing_y), 2) -
        (4 * (spacing_y / max_y) * -y)) ) / (2 * spacing_y / max_y);
  else if (y > (max_y/2) && y < max_y)
    y = (-(1 + 1.5 * spacing_y) + std::sqrt(std::pow((1 + 1.5 * spacing_y), 2) -
        (4 * (-spacing_y / max_y) * (-0.5 * spacing_y * max_y - y))) )/ (2 * -spacing_y / max_y);


  values(0) = x;
  values(1) = y;
}

template <int dim, int spacedim>
double
periodic_hills_pull_back<dim, spacedim>::value(
  const Point<spacedim> &np,
  const unsigned int     component) const
{
  const double max_y = 3.035;
  double       min_y = PeriodicHillsGrid<dim, spacedim>::hill_geometry(
    Point<spacedim>(np[0], 0), alpha, spacing_y)[1];

  double y = (np[1] - min_y) / (1 - min_y / max_y);

  if (y <= (max_y/2))
    y = (-(1 - 0.5 * spacing_y) + std::sqrt(std::pow((1 - 0.5 * spacing_y), 2) -
                                            (4 * (spacing_y / max_y) * -y)) ) / (2 * spacing_y / max_y);
  else if (y > (max_y/2) && y < max_y)
    y = (-(1 + 1.5 * spacing_y) + std::sqrt(std::pow((1 + 1.5 * spacing_y), 2) -
                                            (4 * (-spacing_y / max_y) * (-0.5 * spacing_y * max_y - y))) )/ (2 * -spacing_y / max_y);



  Point<spacedim> op = {np[0], y, np[2]};

  return op[component];
}


template <int dim, int spacedim>
Point<spacedim>
PeriodicHillsGrid<dim, spacedim>::hill_geometry(const Point<spacedim> &p,
                                                double alpha,
                                                double spacing_y)
{
  const double H = 28; // Height dimension to use with polynomials
  double       x = p[0] * H, y = p[1] * H;

  // Polynomial coefficients :
  const double a1 = 2.800000000000E+01,  b1 = 0.000000000000E+00,
               c1 = 6.775070969851E-03,  d1 = -2.124527775800E-03;
  const double a2 = 2.507355893131E+01,  b2 = 9.754803562315E-01,
               c2 = -1.016116352781E-01, d2 = 1.889794677828E-03;
  const double a3 = 2.579601052357E+01,  b3 = +8.206693007457E-01,
               c3 = -9.055370274339E-02, d3 = 1.626510569859E-03;
  const double a4 = 4.046435022819E+01,  b4 = -1.379581654948E+00,
               c4 = 1.945884504128E-02,  d4 = -2.070318932190E-04;
  const double a5 = 1.792461334664E+01,  b5 = +8.743920332081E-01,
               c5 = -5.567361123058E-02, d5 = 6.277731764683E-04;
  const double a6 = 5.639011190988E+01,  b6 = -2.010520359035E+00,
               c6 = 1.644919857549E-02,  d6 = 2.674976141766E-05;

  const double max_y = 3.035 * H;
  double       new_x = (9 * H - x);      // x for the left side of the geometry
  double       pos_y_bottom = 0;
  double       pos_y_top;
  double       pos_y;

  // Gradual spacing and swifting depending on y position
  if (y <= (max_y/2))
  {
    pos_y_bottom = y / -max_y + 0.5;
    y -= spacing_y * pos_y_bottom * y;
  }
  else if (y > max_y/2 && y < max_y)
  {
    pos_y_top = y / max_y - 0.5;
    y += spacing_y * pos_y_top * (max_y - y);
  }

  pos_y = y / -max_y + 1;

  // Polynomial equations :
  if (x >= 0 && x < 9)
  {
    y += pos_y * (a1 + b1 * x + c1 * std::pow(x, 2) + d1 * std::pow(x, 3));
    if (y > 28 && pos_y_bottom == 0.5)
      y = 28;
  }

  else if (x >= 9 && x < 14)
    y += pos_y * (a2 + b2 * x + c2 * std::pow(x, 2) + d2 * std::pow(x, 3));

  else if (x >= 14 && x < 20)
    y += pos_y * (a3 + b3 * x + c3 * std::pow(x, 2) + d3 * std::pow(x, 3));

  else if (x >= 20 && x < 30)
    y += pos_y * (a4 + b4 * x + c4 * std::pow(x, 2) + d4 * std::pow(x, 3));

  else if (x >= 30 && x < 40)
    y += pos_y * (a5 + b5 * x + c5 * std::pow(x, 2) + d5 * std::pow(x, 3));

  else if (x >= 40 && x < 54)
  {
    y += pos_y * (a6 + b6 * x + c6 * std::pow(x, 2) + d6 * std::pow(x, 3));
    if (y < 0)
      y = 0;
  }

  else if (x <= 252 && x >= 243)
  {
    y += pos_y * (a1 + b1 * new_x + c1 * std::pow(new_x, 2) +
                  d1 * std::pow(new_x, 3));
    if (y > 28 && pos_y >= 1)
      y = 28;
  }

  else if (x <= 243 && x > 238)
    y += pos_y *
         (a2 + b2 * new_x + c2 * std::pow(new_x, 2) + d2 * std::pow(new_x, 3));

  else if (x <= 238 && x > 232)
    y += pos_y *
         (a3 + b3 * new_x + c3 * std::pow(new_x, 2) + d3 * std::pow(new_x, 3));

  else if (x <= 232 && x > 222)

    y += pos_y *
         (a4 + b4 * new_x + c4 * std::pow(new_x, 2) + d4 * std::pow(new_x, 3));

  else if (x <= 222 && x > 212)
    y += pos_y *
         (a5 + b5 * new_x + c5 * std::pow(new_x, 2) + d5 * std::pow(new_x, 3));

  else if (x <= 212 && x > 198)
  {
    y += pos_y * (a6 + b6 * new_x + c6 * std::pow(new_x, 2) +
                  d6 * std::pow(new_x, 3));
    if (y < 0)
      y = 0;
  }

  else
    y += 0;

  // Elongation of the geometry with the alpha factor
  // Note : The length of the flat region is always the same length
  double flat_region_length = 5.142 * H;
  double left_hill = 1.929 * H;
  double right_hill = 7.071 * H;

  if (x < left_hill)
    x = alpha * x;
  else if (x > right_hill)
    x = alpha * (x - right_hill) + flat_region_length + alpha  * left_hill;
  else
    x = (x - left_hill) + (alpha * left_hill);

  Point<spacedim> q;
  q[0] = (x / H);
  q[1] = (y / H);

  if (spacedim == 3)
    q[2] = p[2];

  return q;
}

template <int dim, int spacedim>
void
PeriodicHillsGrid<dim, spacedim>::make_grid(
  Triangulation<dim, spacedim> &triangulation)
{
  std::vector<unsigned int> repetitions(2);
  repetitions[0] = repetitions_x;
  repetitions[1] = repetitions_y;

  if (dim == 2)
  {
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              Point<dim>(0.0, 0.0),
                                              Point<dim>(9.0, 3.035),
                                              true);
  }
  else if (dim == 3)
  {
    repetitions.push_back(repetitions_z);
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              repetitions,
                                              Point<dim>(0.0, 0.0, 0.0),
                                              Point<dim>(9.0, 3.035, 4.5),
                                              true);
  }

  // Transformation of the geometry with the hill geometry
  // and gradual shifting of horizontal lines :
  GridTools::transform(
    [this](const Point<spacedim> &p) { return this->hill_geometry(p, alpha, spacing_y); },
    triangulation);

  // Manifold construction
  static const FunctionManifold<dim, spacedim, spacedim> manifold_func(
    std::make_unique<periodic_hills_push_forward<dim, spacedim>>(alpha, spacing_y),
    std::make_unique<periodic_hills_pull_back<dim, spacedim>>(alpha, spacing_y));
  triangulation.set_manifold(1, manifold_func);
  triangulation.set_all_manifold_ids(1);
}


#endif
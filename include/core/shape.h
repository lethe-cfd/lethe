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
 * ---------------------------------------------------------------------*/

#ifndef lethe_shape_h
#define lethe_shape_h

#include <deal.II/base/auto_derivative_function.h>
#include <deal.II/base/function.h>

#if (DEAL_II_VERSION_MAJOR < 10 && DEAL_II_VERSION_MINOR < 4)
#else
#  include <deal.II/base/function_signed_distance.h>
#endif


#include <deal.II/physics/transformations.h>

using namespace dealii;

/**
 * @brief A base class used to represent geometrical entities. Its main uses
 * are to return signed distance and its gradient. It inherits
 * AutoDerivativeFunction so that it can evaluate gradients even when they are
 * not implemented analytically.
 *
 * @tparam dim An integer that denotes the dimension of the space in which
 * the flow is solved
 */
template <int dim>
class Shape : public AutoDerivativeFunction<dim>
{
public:
  /**
   * @brief A general constructor for the Shapes
   */
  Shape(double radius)
    : AutoDerivativeFunction<dim>(1e-8)
    , effective_radius(radius)
  {}

  /**
   * @brief Return the evaluation of the signed distance function of this solid
   * at the given point p
   * Most levelset functions implemented come from Inigo Quilez:
   * iquilezles.org/articles/distfunctions
   */
  virtual double
  value(const Point<dim> & p,
        const unsigned int component = 0) const override = 0;

  /**
   * @brief Return a pointer to a copy of the Shape
   */
  virtual std::shared_ptr<Shape<dim>>
  static_copy() const = 0;

  /**
   * @brief
   * Return the volume displaced by the solid
   *
   */
  virtual double
  displaced_volume(const double fluid_density);

  /**
   * @brief
   * Most value functions assume that the particle's position is at the origin
   * and that the shape is aligned with one of the main axes. This function
   * returns a point that is rotated and translated, in accordance with the
   * current shape position and orientation, so that subsequent calculations for
   * the value function are made more easily; it abstract a step that is
   * required in the value function for most shapes.
   *
   * Returns the centered and aligned point used on the levelset evaluation.
   */
  Point<dim>
  align_and_center(const Point<dim> &evaluation_pt) const;

  // Position of the center of the Shape. It doesn't always correspond to the
  // center of mass
  Point<dim> position;
  // The offset of the center of rotation in relation to the position
  Point<dim> center_of_rotation_offset;
  // The solid orientation, which is defined as the sequential rotation around
  // the axes x->y->z by each of the tensor components, in radian
  Tensor<1, 3> orientation;
  // Effective radius used for crown refinement
  double effective_radius;
};


template <int dim>
class Sphere : public Shape<dim>
{
public:
  Sphere<dim>(double radius)
    : Shape<dim>(radius)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  std::shared_ptr<Shape<dim>>
  static_copy() const override;

  Tensor<1, dim>
  gradient(const Point<dim> & p,
           const unsigned int component = 0) const override;

  double
  displaced_volume(const double fluid_density) override;
};

template <int dim>
class Rectangle : public Shape<dim>
{
public:
  Rectangle<dim>(Tensor<1, 3> half_lengths)
    : Shape<dim>(half_lengths.norm())
    , half_lengths(half_lengths)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  std::shared_ptr<Shape<dim>>
  static_copy() const override;

private:
  Tensor<1, 3> half_lengths;
};

template <int dim>
class Ellipsoid : public Shape<dim>
{
public:
  Ellipsoid<dim>(Tensor<1, 3> radii)
    : Shape<dim>(radii.norm())
    , radii(radii)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  std::shared_ptr<Shape<dim>>
  static_copy() const override;

private:
  Tensor<1, 3> radii;
};

template <int dim>
class Torus : public Shape<dim>
{
public:
  Torus<dim>(double ring_radius, double ring_thickness)
    : Shape<dim>(ring_thickness)
    , ring_radius(ring_radius)
    , ring_thickness(ring_thickness)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  std::shared_ptr<Shape<dim>>
  static_copy() const override;

private:
  double ring_radius;
  double ring_thickness;
};

template <int dim>
class Cone : public Shape<dim>
{
public:
  Cone<dim>(double tan_theta, double height)
    : Shape<dim>(height)
    , tan_theta(tan_theta)
    , height(height)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  std::shared_ptr<Shape<dim>>
  static_copy() const override;

private:
  double tan_theta;
  double height;
};

template <int dim>
class CutHollowSphere : public Shape<dim>
{
public:
  CutHollowSphere<dim>(double r, double h, double t)
    : Shape<dim>(r)
    , r(r)
    , h(h)
    , t(t)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  std::shared_ptr<Shape<dim>>
  static_copy() const override;

private:
  double r;
  double h;
  double t;
};

template <int dim>
class DeathStar : public Shape<dim>
{
public:
  DeathStar<dim>(double ra, double rb, double d)
    : Shape<dim>(ra)
    , ra(ra)
    , rb(rb)
    , d(d)
  {}

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  std::shared_ptr<Shape<dim>>
  static_copy() const override;

private:
  double ra;
  double rb;
  double d;
};

// Composite Shapes are currently used only to output the signed distance of
// particles in the GLS Sharp Navier Stokes solver. The class was however
// designed so that specific composite shapes could be defined through the
// parameter file, although this functionality has not been implemented yet.
template <int dim>
class CompositeShape : public Shape<dim>
{
public:
  CompositeShape<dim>(std::vector<std::shared_ptr<Shape<dim>>> components)
    : Shape<dim>(1.)
    , components(components)
  {
    double radius = 0.;
    for (const std::shared_ptr<Shape<dim>> &elem : components)
      {
        radius = std::max(radius, elem->effective_radius);
      }
    this->effective_radius = radius;
  }

  double
  value(const Point<dim> &p, const unsigned int component = 0) const override;

  std::shared_ptr<Shape<dim>>
  static_copy() const override;

private:
  std::vector<std::shared_ptr<Shape<dim>>> components;
};

#endif // lethe_shape_h
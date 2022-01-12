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
 */

#ifndef lethe_rheological_model_h
#define lethe_rheological_model_h

#include <core/parameters.h>
#include <core/physical_property_model.h>

using namespace dealii;

/**
 * @brief RheologicalModel. Abstract class that allows to calculate the
 * non-newtonian viscosity on each quadrature point and the shear rate
 * magnitude.
 */
template <int dim>
class RheologicalModel : public PhysicalPropertyModel
{
public:
  /**
   * @brief Default constructor
   */
  RheologicalModel()
  {}

  /**
   * @brief Instanciates and returns a pointer to a RheologicalModel object by casting it to
   * the proper child class
   *
   * @param physical_properties Parsed physical properties that will provide
   * either the model rheological model being used or say it is a
   * Newtonian flow
   */
  static std::shared_ptr<RheologicalModel<dim>>
  model_cast(const Parameters::PhysicalProperties &physical_properties);

  /**
   * @brief Returns the magnitude of the shear rate tensor given in parameter.
   *
   * @param shear_rate The shear rate tensor at the position of the
   * considered quadrature point
   */
  double
  get_shear_rate_magnitude(const Tensor<2, dim> shear_rate);
};

template <int dim>
class Newtonian : public RheologicalModel<dim>
{
public:
  /**
   * @brief Parameter constructor
   *
   * @param viscosity The constant newtonian viscosity
   */
  Newtonian(const double &p_viscosity)
    : viscosity(p_viscosity)
  {}

  /**
   * @brief Returns the viscosity.
   *
   * @param field_values Values of the field on which the viscosity may depend on.
   * For the constant viscosity, the viscosity does not depend on anything.
   */
  double
  value(const std::map<field, double> & /*field_values*/) override;

  /**
   * @brief vector_value Calculates the vector values of the viscosity.
   * @param field_vectors Values of the field on which the viscosity may depend on. These are not used for the constant viscosity
   */
  void
  vector_value(const std::map<field, std::vector<double>> & /*field_vectors*/,
               std::vector<double> &property_vector) override
  {
    std::fill(property_vector.begin(), property_vector.end(), viscosity);
  }

  /**
   * @brief jacobian Calculates the jacobian (the partial derivative) of the viscosity
   * with respect to a field
   * @param field_values Value of the various fields on which the property may depend. The constant viscosity does not depend on anything
   * @param id Indicator of the field with respect to which the jacobian
   * should be calculated
   * @return value of the partial derivative of the viscosity with respect to the field.
   */

  double
  jacobian(const std::map<field, double> & /*field_values*/,
           field /*id*/) override
  {
    return 0;
  };

  /**
   * @brief vector_jacobian Calculate the derivative of the with respect to a field
   * @param field_vectors Vector for the values of the fields used to evaluate the property
   * @param id Identifier of the field with respect to which a derivative should be calculated
   * @param jacobian Vector of the value of the derivative of the viscosity with respect to the field id
   */

  void
  vector_jacobian(
    const std::map<field, std::vector<double>> & /*field_vectors*/,
    const field /*id*/,
    std::vector<double> &jacobian_vector) override
  {
    std::fill(jacobian_vector.begin(), jacobian_vector.end(), 0);
  }

private:
  double viscosity;
};

template <int dim>
class PowerLaw : public RheologicalModel<dim>
{
public:
  /**
   * @brief Parameter constructor
   *
   * @param non_newtonian_parameters The non newtonian parameters
   */
  PowerLaw(const double K, const double n, const double shear_rate_min)
    : K(K)
    , n(n)
    , shear_rate_min(shear_rate_min)
  {
    this->model_depends_on[shear_rate] = false;
  }

  /**
   * @brief Returns the non-newtonian viscosity.
   *
   * @param field_values The values Values of the field on which the viscosity may depend on. For this model, it only depends on the magnitude of the shear rate tensor
   *
   * Source : Morrison, F. A. (2001). No Memory: Generalized Newtonian Fluids.
   * Understanding Rheology. Raymond F. Boyer Librabry Collection, Oxford
   * University Press.
   */
  double
  value(const std::map<field, double> &field_values) override;

  /**
   * @brief vector_value Calculates the vector values of the viscosity.
   * @param field_vectors Values of the field on which the viscosity may depend on. The power-law viscosity depends on the shear rate.
   */
  void
  vector_value(const std::map<field, std::vector<double>> & /*field_vectors*/,
               std::vector<double> &property_vector) override;

  /**
   * @brief jacobian Calculates the jacobian (the partial derivative) of the viscosity
   * with respect to a field
   * @param field_values Value of the various fields on which the property may depend. The constant viscosity does not depend on anything
   * @param id Indicator of the field with respect to which the jacobian
   * should be calculated
   * @return value of the partial derivative of the viscosity with respect to the field.
   */

  double
  jacobian(const std::map<field, double> & /*field_values*/,
           field /*id*/) override;

  /**
   * @brief vector_jacobian Calculate the derivative of the with respect to a field
   * @param field_vectors Vector for the values of the fields used to evaluate the property
   * @param id Identifier of the field with respect to which a derivative should be calculated
   * @param jacobian Vector of the value of the derivative of the viscosity with respect to the field id
   */

  void
  vector_jacobian(
    const std::map<field, std::vector<double>> & /*field_vectors*/,
    const field /*id*/,
    std::vector<double> &jacobian_vector) override;

private:
  inline double
  calculate_viscosity(const double shear_rate_magnitude)
  {
    return shear_rate_magnitude > shear_rate_min ?
             K * std::pow(shear_rate_magnitude, n - 1) :
             K * std::pow(shear_rate_min, n - 1);
  }

  inline double
  calculate_derivative(const double shear_rate_magnitude)
  {
    return shear_rate_magnitude > shear_rate_min ?
             (n - 1) * K * std::pow(shear_rate_magnitude, n - 2) :
             0;
  }


  const double K;
  const double n;
  const double shear_rate_min;
};

template <int dim>
class Carreau : public RheologicalModel<dim>
{
public:
  /**
   * @brief Parameter constructor
   *
   * @param non_newtonian_parameters The non newtonian parameters
   */
  Carreau(const double viscosity_0,
          const double viscosity_inf,
          const double lambda,
          const double a,
          const double n)
    : viscosity_0(viscosity_0)
    , viscosity_inf(viscosity_inf)
    , lambda(lambda)
    , a(a)
    , n(n)
  {
    this->model_depends_on[shear_rate] = false;
  }

  /**
   * @brief Returns the non-newtonian viscosity.
   *
   * @param field_values The values of the field on which the viscosity may depend on. For this model, it only depends on the magnitude of the shear rate tensor
   *
   * Source : Morrison, F. A. (2001). No Memory: Generalized Newtonian Fluids.
   * Understanding Rheology. Raymond F. Boyer Librabry Collection, Oxford
   * University Press.
   */
  double
  value(const std::map<field, double> & /*field_values*/) override;

  /**
   * @brief vector_value Calculates the vector values of the viscosity.
   * @param field_vectors Values of the field on which the viscosity may depend on. The power-law viscosity depends on the shear rate.
   */
  void
  vector_value(const std::map<field, std::vector<double>> & /*field_vectors*/,
               std::vector<double> &property_vector) override;

  /**
   * @brief jacobian Calculates the jacobian (the partial derivative) of the viscosity
   * with respect to a field
   * @param field_values Value of the various fields on which the property may depend. The constant viscosity does not depend on anything
   * @param id Indicator of the field with respect to which the jacobian
   * should be calculated
   * @return value of the partial derivative of the viscosity with respect to the field.
   */
  double
  jacobian(const std::map<field, double> & /*field_values*/,
           field /*id*/) override;

  /**
   * @brief vector_jacobian Calculate the derivative of the with respect to a field
   * @param field_vectors Vector for the values of the fields used to evaluate the property
   * @param id Identifier of the field with respect to which a derivative should be calculated
   * @param jacobian Vector of the value of the derivative of the viscosity with respect to the field id
   */

  void
  vector_jacobian(
    const std::map<field, std::vector<double>> & /*field_vectors*/,
    const field /*id*/,
    std::vector<double> &jacobian_vector) override;

private:
  inline double
  calculate_viscosity(const double shear_rate_magnitude)
  {
    return viscosity_inf +
           (viscosity_0 - viscosity_inf) *
             std::pow(1.0 + std::pow(shear_rate_magnitude * lambda, a),
                      (n - 1) / a);
  }

  double viscosity_0;
  double viscosity_inf;
  double lambda;
  double a;
  double n;
};


#endif

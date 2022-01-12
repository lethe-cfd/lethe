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

#ifndef lethe_thermal_conductivity_model_h
#define lethe_thermal_conductivity_model_h

#include <core/parameters.h>
#include <core/physical_property_model.h>

/**
 * @brief ThermalConductivityModel. Abstract class that allows to calculate the
 * thermal conductivity on each quadrature point using the temperature of the
 * fluid.
 */
class ThermalConductivityModel : public PhysicalPropertyModel
{
public:
};


/**
 * @brief Constant thermal conductivity.
 */
class ThermalConductivityConstant : public ThermalConductivityModel
{
public:
  /**
   * @brief Default constructor
   */
  ThermalConductivityConstant(const double p_thermal_conductivity)
    : thermal_conductivity(p_thermal_conductivity)
  {}

  /**
   * @brief value Calculates the value the thermal conductivity
   * @param fields_value Value of the various field on which the thermal conductivity depends.
   * @return value of the thermal conductivity calculated with the fields_value.
   */
  virtual double
  value(const std::map<field, double> & /*fields_value*/) override
  {
    return thermal_conductivity;
  };

  /**
   * @brief vector_value Calculates the vector value of thermal conductivities
   * @param field_vectors Vector of properties on which the thermal conductivities depend
   * @param property_vector Values of the thermal conductivities
   */
  virtual void
  vector_value(const std::map<field, std::vector<double>> & /*field_vectors*/,
               std::vector<double> &property_vector) override
  {
    property_vector.assign(property_vector.size(), thermal_conductivity);
  }

  /**
   * @brief jacobian Calcualtes the jacobian (the partial derivative) of the thermal conductivity with respect to a field
   * @param field_values Value of the various fields on which the property may depend.
   * @param id Indicator of the field with respect to which the jacobian
   * should be calculated
   * @return value of the partial derivative of the thermal conductivity with respect to the field.
   */

  virtual double
  jacobian(const std::map<field, double> & /*field_values*/,
           field /*id*/) override
  {
    return 0;
  };

  /**
   * @brief vector_jacobian Calculate the derivative of the thermal conductivity with respect to a field
   * @param field_vectors Vector for the values of the fields used to evaluate the property
   * @param id Identifier of the field with respect to which a derivative should be calculated
   * @param jacobian Vector of the value of the derivative of the thermal conductivity with respect to the field id
   */

  virtual void
  vector_jacobian(
    const std::map<field, std::vector<double>> & /*field_vectors*/,
    const field /*id*/,
    std::vector<double> &jacobian_vector) override
  {
    std::fill(jacobian_vector.begin(), jacobian_vector.end(), 0);
  };

private:
  const double thermal_conductivity;
};


/**
 * @brief ThermalConductivityLinear Implements a linear temperature-dependant thermal conductivity of the form k = A + BT
 */
class ThermalConductivityLinear : public ThermalConductivityModel
{
public:
  /**
   * @brief Default constructor
   */
  ThermalConductivityLinear(const double A, const double B)
    : A(A)
    , B(B)
  {
    this->model_depends_on[field::temperature] = true;
  }

  /**
   * @brief value Calculates the value the thermal conductivity
   * @param fields_value Value of the various field on which the thermal conductivity depends.
   * @return value of the thermal conductivity calculated with the fields_value.
   */
  virtual double
  value(const std::map<field, double> &fields_value) override
  {
    return A + B * fields_value.at(field::temperature);
  };

  /**
   * @brief vector_value Calculates the vector value of thermal conductivities
   * @param field_vectors Vector of properties on which the thermal conductivities depend
   * @param property_vector Values of the thermal conductivities
   */
  virtual void
  vector_value(const std::map<field, std::vector<double>> &field_vectors,
               std::vector<double> &property_vector) override
  {
    const std::vector<double> &T = field_vectors.at(field::temperature);
    for (unsigned int i = 0; i < property_vector.size(); ++i)
      property_vector[i] = A + B * T[i];
  }

  /**
   * @brief jacobian Calcualtes the jacobian (the partial derivative) of the thermal conductivity with respect to a field
   * @param field_values Value of the various fields on which the property may depend.
   * @param id Indicator of the field with respect to which the jacobian
   * should be calculated
   * @return value of the partial derivative of the thermal conductivity with respect to the field.
   */

  virtual double
  jacobian(const std::map<field, double> & /*field_values*/,
           field /*id*/) override
  {
    return B;
  };

  /**
   * @brief vector_jacobian Calculate the derivative of the thermal conductivity with respect to a field
   * @param field_vectors Vector for the values of the fields used to evaluate the property
   * @param id Identifier of the field with respect to which a derivative should be calculated
   * @param jacobian Vector of the value of the derivative of the thermal conductivity with respect to the field id
   */

  virtual void
  vector_jacobian(
    const std::map<field, std::vector<double>> & /*field_vectors*/,
    const field /*id*/,
    std::vector<double> &jacobian_vector) override
  {
    std::fill(jacobian_vector.begin(), jacobian_vector.end(), B);
  };

private:
  const double A;
  const double B;
};

#endif

#include <core/shape_parsing.h>

template <int dim>
std::shared_ptr<Shape<dim>>
ShapeGenerator::initialize_shape(const std::string   type,
                                 const std::string   shape_arguments_str,
                                 const Point<dim> &  position,
                                 const Tensor<1, 3> &orientation)
{
  std::shared_ptr<Shape<dim>> shape;
  std::vector<double>         shape_arguments;
  if (type == "rbf" || type == "composite")
    {
      shape = initialize_shape_from_file(type,
                                         shape_arguments_str,
                                         position,
                                         orientation);
    }
  else
    {
      std::vector<std::string> shape_arguments_str_list(
        Utilities::split_string_list(shape_arguments_str, ";"));
      shape_arguments = Utilities::string_to_double(shape_arguments_str_list);
      shape           = initialize_shape_from_vector(type,
                                           shape_arguments,
                                           position,
                                           orientation);
    }
  return shape;
}

template <int dim>
std::shared_ptr<Shape<dim>>
ShapeGenerator::initialize_shape_from_vector(
  const std::string         type,
  const std::vector<double> shape_arguments,
  const Point<dim> &        position,
  const Tensor<1, 3> &      orientation)
{
  std::shared_ptr<Shape<dim>> shape;
  if (type == "sphere")
    shape =
      std::make_shared<Sphere<dim>>(shape_arguments[0], position, orientation);
  else if (type == "rectangle")
    {
      Tensor<1, dim> half_lengths;
      for (unsigned int i = 0; i < dim; ++i)
        {
          half_lengths[i] = shape_arguments[i];
        }
      shape =
        std::make_shared<Rectangle<dim>>(half_lengths, position, orientation);
    }
  else if (type == "ellipsoid")
    {
      Tensor<1, dim> radii;
      for (unsigned int i = 0; i < dim; ++i)
        {
          radii[i] = shape_arguments[i];
        }
      shape = std::make_shared<Ellipsoid<dim>>(radii, position, orientation);
    }
  else if (type == "torus")
    {
      if constexpr (dim == 3)
        shape = std::make_shared<Torus<dim>>(shape_arguments[0],
                                             shape_arguments[1],
                                             position,
                                             orientation);
    }
  else if (type == "cone")
    {
      if constexpr (dim == 3)
        shape = std::make_shared<Cone<dim>>(shape_arguments[0],
                                            shape_arguments[1],
                                            position,
                                            orientation);
    }
  else if (type == "cylinder")
    {
      if constexpr (dim == 3)
        shape = std::make_shared<Cylinder<dim>>(shape_arguments[0],
                                                shape_arguments[1],
                                                position,
                                                orientation);
    }
  else if (type == "cylindrical tube")
    {
      if constexpr (dim == 3)
        shape = std::make_shared<CylindricalTube<dim>>(shape_arguments[0],
                                                       shape_arguments[1],
                                                       shape_arguments[2],
                                                       position,
                                                       orientation);
    }
  else if (type == "cylindrical helix")
    {
      if constexpr (dim == 3)
        shape = std::make_shared<CylindricalHelix<dim>>(shape_arguments[0],
                                                        shape_arguments[1],
                                                        shape_arguments[2],
                                                        shape_arguments[3],
                                                        position,
                                                        orientation);
    }
  else if (type == "cut hollow sphere")
    {
      if constexpr (dim == 3)
        shape = std::make_shared<CutHollowSphere<dim>>(shape_arguments[0],
                                                       shape_arguments[1],
                                                       shape_arguments[2],
                                                       position,
                                                       orientation);
    }
  else if (type == "death star")
    {
      if constexpr (dim == 3)
        shape = std::make_shared<DeathStar<dim>>(shape_arguments[0],
                                                 shape_arguments[1],
                                                 shape_arguments[2],
                                                 position,
                                                 orientation);
    }
  else if (type == "rbf")
    {
      shape =
        std::make_shared<RBFShape<dim>>(shape_arguments, position, orientation);
    }
  else
    StandardExceptions::ExcNotImplemented();
  return shape;
}

template <int dim>
std::shared_ptr<Shape<dim>>
ShapeGenerator::initialize_shape_from_file(const std::string   type,
                                           const std::string   file_name,
                                           const Point<dim> &  position,
                                           const Tensor<1, 3> &orientation)
{
  std::shared_ptr<Shape<dim>> shape;
  std::vector<double>         shape_arguments;
  if (type == "rbf")
    {
      if (file_name == "1") // Default case
        {
          // Default weight, support radius, basis function, x, y
          shape_arguments = {2.0, 1.0, 2.0, 0.0, 0.0};
          if constexpr (dim == 3)
            // and z
            shape_arguments.insert(shape_arguments.end(), 0.0);
        }
      else
        {
          // The following lines retrieve information regarding an RBF
          // with a given file name. Then, it converts the information
          // into one vector which is used to initialize the RBF shape.
          // All the information is concatenated into only one object so
          // that the usual initialization function can be called.
          std::map<std::string, std::vector<double>> rbf_data;
          fill_vectors_from_file(rbf_data, file_name, " ");
          size_t number_of_nodes = rbf_data["weight"].size();
          shape_arguments.reserve((dim + 3) * number_of_nodes);
          shape_arguments.insert(shape_arguments.end(),
                                 rbf_data["weight"].begin(),
                                 rbf_data["weight"].end());
          shape_arguments.insert(shape_arguments.end(),
                                 rbf_data["support_radius"].begin(),
                                 rbf_data["support_radius"].end());
          shape_arguments.insert(shape_arguments.end(),
                                 rbf_data["basis_function"].begin(),
                                 rbf_data["basis_function"].end());
          shape_arguments.insert(shape_arguments.end(),
                                 rbf_data["node_x"].begin(),
                                 rbf_data["node_x"].end());
          shape_arguments.insert(shape_arguments.end(),
                                 rbf_data["node_y"].begin(),
                                 rbf_data["node_y"].end());
          shape_arguments.insert(shape_arguments.end(),
                                 rbf_data["node_z"].begin(),
                                 rbf_data["node_z"].end());
        }
      shape =
        std::make_shared<RBFShape<dim>>(shape_arguments, position, orientation);
    }
  else if (type == "composite")
    {
      // The following lines retrieve information regarding a
      // composite shape.
      std::map<unsigned int, std::shared_ptr<Shape<dim>>> components;
      std::map<unsigned int,
               std::tuple<typename CompositeShape<dim>::BooleanOperation,
                          unsigned int,
                          unsigned int>>
        operations;
      // In the file, we first loop over all component shapes, then
      // we loop over operations
      std::ifstream myfile(file_name);
      // Read file line by line for section names or arguments
      if (myfile)
        {
          std::string              line;
          std::vector<std::string> column_names;
          std::vector<double>      line_of_data;
          bool                     parsing_shapes     = false;
          bool                     parsing_operations = false;
          while (std::getline(myfile, line))
            {
              if (line == "")
                continue;
              if (line == "shapes")
                {
                  parsing_shapes     = true;
                  parsing_operations = false;
                }
              else if (line == "operations")
                {
                  parsing_shapes     = false;
                  parsing_operations = true;
                }
              else
                {
                  std::vector<std::string> list_of_words_base =
                    Utilities::split_string_list(line, ";");
                  std::vector<std::string> list_of_words_clean;
                  for (unsigned int j = 0; j < list_of_words_base.size(); ++j)
                    {
                      if (list_of_words_base[j] != "")
                        {
                          list_of_words_clean.push_back(list_of_words_base[j]);
                        }
                    }
                  if (parsing_shapes)
                    {
                      unsigned int identifier    = stoi(list_of_words_clean[0]);
                      std::string  type          = list_of_words_clean[1];
                      std::string  arguments_str = list_of_words_clean[2];
                      std::string  position_str  = list_of_words_clean[3];
                      std::string  orientation_str = list_of_words_clean[4];

                      std::vector<std::string> arguments_str_component =
                        Utilities::split_string_list(arguments_str, ":");
                      std::vector<std::string> position_str_component =
                        Utilities::split_string_list(position_str, ":");
                      std::vector<std::string> orientation_str_component =
                        Utilities::split_string_list(orientation_str, ":");

                      shape_arguments =
                        Utilities::string_to_double(arguments_str_component);
                      std::vector<double> temp_position_vec =
                        Utilities::string_to_double(position_str_component);
                      std::vector<double> temp_orientation_vec =
                        Utilities::string_to_double(orientation_str_component);

                      Point<dim> temp_position;
                      Point<3>   temp_orientation =
                        Point<3>({temp_orientation_vec[0],
                                  temp_orientation_vec[1],
                                  temp_orientation_vec[2]});
                      temp_position[0] = temp_position_vec[0];
                      temp_position[1] = temp_position_vec[1];
                      if constexpr (dim == 3)
                        temp_position[2] = temp_position_vec[2];

                      std::shared_ptr<Shape<dim>> shape_temp;
                      shape_temp = ShapeGenerator::initialize_shape_from_vector(
                        type, shape_arguments, Point<dim>(), Tensor<1, 3>());
                      shape_temp->set_position(temp_position);
                      shape_temp->set_orientation(temp_orientation);
                      components[identifier] = shape_temp->static_copy();
                    }
                  else if (parsing_operations)
                    {
                      unsigned int identifier    = stoi(list_of_words_clean[0]);
                      std::string  type          = list_of_words_clean[1];
                      std::string  arguments_str = list_of_words_clean[2];
                      std::vector<std::string> arguments_str_component =
                        Utilities::split_string_list(arguments_str, ":");

                      unsigned int first_shape =
                        stoi(arguments_str_component[0]);
                      unsigned int second_shape =
                        stoi(arguments_str_component[1]);
                      if (type == "union")
                        {
                          operations[identifier] = std::make_tuple(
                            CompositeShape<dim>::BooleanOperation::Union,
                            first_shape,
                            second_shape);
                        }
                      else if (type == "difference")
                        {
                          operations[identifier] = std::make_tuple(
                            CompositeShape<dim>::BooleanOperation::Difference,
                            first_shape,
                            second_shape);
                        }
                      else if (type == "intersection")
                        {
                          operations[identifier] = std::make_tuple(
                            CompositeShape<dim>::BooleanOperation::Intersection,
                            first_shape,
                            second_shape);
                        }
                    }
                }
            }
          myfile.close();
          shape = std::make_shared<CompositeShape<dim>>(components,
                                                        operations,
                                                        position,
                                                        orientation);
        }
      else
        throw std::invalid_argument(file_name);
    }
  return shape;
}

template std::shared_ptr<Shape<2>>
ShapeGenerator::initialize_shape(const std::string   type,
                                 const std::string   arguments,
                                 const Point<2> &    position,
                                 const Tensor<1, 3> &orientation);
template std::shared_ptr<Shape<3>>
ShapeGenerator::initialize_shape(const std::string   type,
                                 const std::string   arguments,
                                 const Point<3> &    position,
                                 const Tensor<1, 3> &orientation);
template std::shared_ptr<Shape<2>>
ShapeGenerator::initialize_shape_from_vector(
  const std::string         type,
  const std::vector<double> shape_arguments,
  const Point<2> &          position,
  const Tensor<1, 3> &      orientation);
template std::shared_ptr<Shape<3>>
ShapeGenerator::initialize_shape_from_vector(
  const std::string         type,
  const std::vector<double> shape_arguments,
  const Point<3> &          position,
  const Tensor<1, 3> &      orientation);
template std::shared_ptr<Shape<2>>
ShapeGenerator::initialize_shape_from_file(const std::string   type,
                                           const std::string   file_name,
                                           const Point<2> &    position,
                                           const Tensor<1, 3> &orientation);
template std::shared_ptr<Shape<3>>
ShapeGenerator::initialize_shape_from_file(const std::string   type,
                                           const std::string   file_name,
                                           const Point<3> &    position,
                                           const Tensor<1, 3> &orientation);
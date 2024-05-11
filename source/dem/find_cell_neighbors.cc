#include <dem/find_cell_neighbors.h>

using namespace dealii;

// The constructor of this class is empty
template <int dim>
FindCellNeighbors<dim>::FindCellNeighbors()
{}

// This function finds the neighbor list (without repetition) of all the active
// cells in the triangulation
template <int dim>
void
FindCellNeighbors<dim>::find_cell_neighbors(
  const parallel::distributed::Triangulation<dim> &triangulation,
  typename DEM::dem_data_structures<dim>::cells_neighbor_list
    &cells_local_neighbor_list,
  typename DEM::dem_data_structures<dim>::cells_neighbor_list
    &cells_ghost_neighbor_list)
{
  // The output vectors of the function are cells_local_neighbor_list and
  // cells_ghost_neighbor_list. The first contains all the local neighbors cells
  // of all local cells; while the second contains all the ghost cells of all
  // local cells. They are two vectors with size of the number of active cells.
  // The first elements of all vectors are the main cells
  typename DEM::dem_data_structures<dim>::cell_vector local_neighbor_vector;
  typename DEM::dem_data_structures<dim>::cell_vector ghost_neighbor_vector;

  // This vector is used to avoid repetition of adjacent cells. For instance if
  // cell B is recognized as the neighbor of cell A, cell A will not be added to
  // the neighbor list of cell B again. This is done using the total_cell_list
  // vector
  typename DEM::dem_data_structures<dim>::cell_set total_cell_list;

  // For each cell, the cell vertices are found and used to find adjacent cells.
  // The reason is to find the cells located on the corners of the main cell.
  auto v_to_c = GridTools::vertex_to_cell_map(triangulation);

  // Looping over cells
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      // If the cell is owned by the processor
      if (cell->is_locally_owned())
        {
          // The first element of each vector is the cell itself.
          local_neighbor_vector.push_back(cell);

          total_cell_list.insert(cell);

          for (unsigned int vertex = 0; vertex < cell->n_vertices(); ++vertex)
            {
              for (const auto &neighbor : v_to_c[cell->vertex_index(vertex)])
                {
                  if (neighbor->is_locally_owned())
                    {
                      // Check if the neighbor cell was already treated as the
                      // main cell. This makes sure that there is no repetion.
                      auto search_iterator = total_cell_list.find(neighbor);

                      // Check if the neighbor cell is already in the neighbor
                      // list of the main cell. This can happen when cells share
                      // more than one vertex.
                      auto local_search_iterator =
                        std::find(local_neighbor_vector.begin(),
                                  local_neighbor_vector.end(),
                                  neighbor);

                      // If the cell (neighbor) is a local cell and not present
                      // in the total_cell_list vector, it will be added as the
                      // neighbor of the main cell
                      // ("cell") and also to the total_cell_list to avoid
                      // repetition for next cells.

                      // If the neighboring cell has not yet been treated as a
                      // main cell and is still not in the local search list, it
                      // will be added.
                      if (search_iterator == total_cell_list.end() &&
                          local_search_iterator == local_neighbor_vector.end())
                        {
                          local_neighbor_vector.push_back(neighbor);
                        }
                    }
                  // If the neighbor cell is a ghost, it should be added in
                  // the ghost_neighbor_vector container
                  else if (neighbor->is_ghost())
                    {
                      auto ghost_search_iterator =
                        std::find(ghost_neighbor_vector.begin(),
                                  ghost_neighbor_vector.end(),
                                  neighbor);
                      if (ghost_search_iterator == ghost_neighbor_vector.end())
                        {
                          // Even though the main cell is not a ghost cell, the
                          // first iterator of the ghost_neighbor_vector most be
                          // the main cell. For this reason, we check is the
                          // ghost_neighbor_vector is empty.
                          if (ghost_neighbor_vector.empty())
                            {
                              ghost_neighbor_vector.push_back(cell);
                            }
                          ghost_neighbor_vector.push_back(neighbor);
                        }
                    }
                }
            }
        }
      if (!local_neighbor_vector.empty())
        cells_local_neighbor_list.push_back(local_neighbor_vector);
      if (!ghost_neighbor_vector.empty())
        cells_ghost_neighbor_list.push_back(ghost_neighbor_vector);

      local_neighbor_vector.clear();
      ghost_neighbor_vector.clear();
    }
}

template <int dim>
void
FindCellNeighbors<dim>::find_cell_periodic_neighbors(
  const parallel::distributed::Triangulation<dim> &triangulation,
  const typename DEM::dem_data_structures<dim>::periodic_boundaries_cells_info
    &periodic_boundaries_cells_information,
  typename DEM::dem_data_structures<dim>::cells_neighbor_list
    &cells_local_periodic_neighbor_list,
  typename DEM::dem_data_structures<dim>::cells_neighbor_list
    &cells_ghost_periodic_neighbor_list,
  typename DEM::dem_data_structures<dim>::cells_neighbor_list
    &cells_ghost_local_periodic_neighbor_list)
{
  typename DEM::dem_data_structures<dim>::cell_vector
    local_periodic_neighbor_vector;
  // TODO local_periodic_neighbor_vector.reserve(9) this vector?
  typename DEM::dem_data_structures<dim>::cell_vector
    ghost_periodic_neighbor_vector;
  typename DEM::dem_data_structures<dim>::cell_vector
    ghost_local_periodic_neighbor_vector;

  typename DEM::dem_data_structures<dim>::cell_set total_cell_list;
  typename DEM::dem_data_structures<dim>::cell_set total_ghost_cell_list;

  // For each cell, the cell vertices are found and used to find adjacent cells.
  // The reason is to find the cells located on the corners of the main cell
  // (sharing 1 or 2 vertices).
  auto v_to_c = GridTools::vertex_to_cell_map(triangulation);
  // TODO v_to_c.reserve(8) this vector?

  // A map of coinciding vertices labeled by an arbitrary element from them
  std::map<unsigned int, std::vector<unsigned int>> coinciding_vertex_groups;

  // A map of vertex to the label of a group of coinciding vertices it is part
  // of.
  std::map<unsigned int, unsigned int> vertex_to_coinciding_vertex_group;

  // Collect for a given triangulation all locally relevant vertices that
  // coincide to periodicity.
  GridTools::collect_coinciding_vertices(triangulation,
                                         coinciding_vertex_groups,
                                         vertex_to_coinciding_vertex_group);

  // Looping over cells struct at periodic boundaries 0
  for (const auto &pb_cell_struct : periodic_boundaries_cells_information)
    {
      auto &cell = pb_cell_struct.second.cell;

      // If the cell is owned by the processor
      if (cell->is_locally_owned())
        {
          // The first element of each vector is the cell itself.
          local_periodic_neighbor_vector.push_back(cell);
          total_cell_list.insert(cell);

          // Empty list of periodic cell neighbor
          typename DEM::dem_data_structures<dim>::cell_vector
            periodic_neighbor_list;

          // Get the periodic neighbor of the cell
          get_periodic_neighbor_list(cell,
                                     coinciding_vertex_groups,
                                     vertex_to_coinciding_vertex_group,
                                     v_to_c,
                                     periodic_neighbor_list);

          // Loop over the periodic neighbor cells of the main cell
          for (const auto &periodic_neighbor : periodic_neighbor_list)
            {
              if (periodic_neighbor->is_locally_owned())
                {
                  // A periodic neighbor cell can't be the main cell, thus we
                  // don't need to check if it is in the total_cell_list.

                  // auto search_iterator =
                  // total_cell_list.find(periodic_neighbor);

                  // periodic_neighbor_list may have duplicates iterator. We
                  // need to check if the neighbor cell is not already in the
                  // list.
                  auto local_search_iterator =
                    std::find(local_periodic_neighbor_vector.begin(),
                              local_periodic_neighbor_vector.end(),
                              periodic_neighbor);

                  // If the cell neighbor is a local cell and is not already in
                  // the local_search_iterator list, it will be added to it.
                  if (local_search_iterator ==
                      local_periodic_neighbor_vector.end())
                    {
                      local_periodic_neighbor_vector.push_back(
                        periodic_neighbor);
                    }
                }
              else if (periodic_neighbor->is_ghost())
                {
                  // periodic_neighbor_list may have duplicates iterator. We
                  // need to check if the neighbor cell is not already in the
                  // list.
                  auto ghost_search_iterator =
                    std::find(ghost_periodic_neighbor_vector.begin(),
                              ghost_periodic_neighbor_vector.end(),
                              periodic_neighbor);

                  // If the cell neighbor is a ghost cell and is not already in
                  // the ghost_search_iterator list, it will be added to it.
                  if (ghost_search_iterator ==
                      ghost_periodic_neighbor_vector.end())
                    {
                      // The first element of each vector is the cell itself.
                      if (ghost_periodic_neighbor_vector.empty())
                        {
                          ghost_periodic_neighbor_vector.push_back(cell);
                        }

                      ghost_periodic_neighbor_vector.push_back(
                        periodic_neighbor);
                    }
                }
            }

          // local_periodic_neighbor_vector will never be empty in this if
          // (minimum size of 1). See line 167 to 170. Either remove this "if"
          // or : if(local_periodic_neighbor_vector.size() > 1)
          if (!local_periodic_neighbor_vector.empty())
            cells_local_periodic_neighbor_list.push_back(
              local_periodic_neighbor_vector);
          if (!ghost_periodic_neighbor_vector.empty())
            cells_ghost_periodic_neighbor_list.push_back(
              ghost_periodic_neighbor_vector);
          local_periodic_neighbor_vector.clear();
          ghost_periodic_neighbor_vector.clear();
        }
      else if (cell->is_ghost())
        {
          // Since periodic cells are mapped on one side only (cells on pb 0
          // with cells on pb 1), we need a 3rd container for ghost-local
          // contacts for force calculation. Here we store local neighbors of
          // ghost cells.

          // The first element of each vector is the cell itself
          ghost_local_periodic_neighbor_vector.push_back(cell);
          total_ghost_cell_list.insert(cell);

          // Empty list of periodic cell neighbor
          typename DEM::dem_data_structures<dim>::cell_vector
            periodic_neighbor_list;

          // Get the periodic neighbor of the cell
          get_periodic_neighbor_list(cell,
                                     coinciding_vertex_groups,
                                     vertex_to_coinciding_vertex_group,
                                     v_to_c,
                                     periodic_neighbor_list);

          // Loop over the periodic neighbor cells of the main cell
          for (const auto &periodic_neighbor : periodic_neighbor_list)
            {
              // We only check if the neighbor cell is locally owned, since
              // ghost-ghost is not a thing.
              if (periodic_neighbor->is_locally_owned())
                {
                  // A periodic neighbor cell can't be the main cell, thus we
                  // don't need to check if it is in the total_cell_list.

                  // periodic_neighbor_list may have duplicates iterator. We
                  // need to check if the neighbor cell is not already in the
                  // list.
                  auto local_search_iterator =
                    std::find(ghost_local_periodic_neighbor_vector.begin(),
                              ghost_local_periodic_neighbor_vector.end(),
                              periodic_neighbor);

                  if (local_search_iterator ==
                      ghost_local_periodic_neighbor_vector.end())
                    {
                      ghost_local_periodic_neighbor_vector.push_back(
                        periodic_neighbor);
                    }
                }
            }
          // ghost_local_periodic_neighbor_vector will never be empty in this if
          // (minimum size of 1). See line 257 to 260. Either remove this "if"
          // or : if(ghost_local_periodic_neighbor_vector.size() > 1)
          if (!ghost_local_periodic_neighbor_vector.empty())
            cells_ghost_local_periodic_neighbor_list.push_back(
              ghost_local_periodic_neighbor_vector);
          ghost_local_periodic_neighbor_vector.clear();
        }
    }
}

// This function finds the full neighbor list (with repetition) of all the
// active cells in the triangulation. Because particle-floating mesh
// contacts need all the particles located in ALL the neighbor cells of
// the main cell to search for possible collisions with the floating mesh.
template <int dim>
void
FindCellNeighbors<dim>::find_full_cell_neighbors(
  const parallel::distributed::Triangulation<dim> &triangulation,
  typename DEM::dem_data_structures<dim>::cells_total_neighbor_list
    &cells_total_neighbor_list)
{
  auto v_to_c = GridTools::vertex_to_cell_map(triangulation);

  // Looping over cells
  for (const auto &cell : triangulation.active_cell_iterators())
    {
      // If the cell is owned by the processor
      if (cell->is_locally_owned())
        {
          std::vector<typename Triangulation<dim>::active_cell_iterator>
            full_neighbor_vector(0);

          full_neighbor_vector.push_back(cell);

          for (unsigned int vertex = 0; vertex < cell->n_vertices(); ++vertex)
            {
              for (const auto &neighbor : v_to_c[cell->vertex_index(vertex)])
                {
                  if (neighbor->is_locally_owned())
                    {
                      auto search_iterator =
                        std::find(full_neighbor_vector.begin(),
                                  full_neighbor_vector.end(),
                                  neighbor);

                      if (search_iterator == full_neighbor_vector.end())
                        {
                          full_neighbor_vector.push_back(neighbor);
                        }
                    }
                }
            }

          cells_total_neighbor_list.insert(
            {cell->global_active_cell_index(), full_neighbor_vector});
        }
    }
}
template <int dim>
void
FindCellNeighbors<dim>::get_periodic_neighbor_list(
  const typename Triangulation<dim>::active_cell_iterator &cell,
  const std::map<unsigned int, std::vector<unsigned int>>
                                             &coinciding_vertex_groups,
  const std::map<unsigned int, unsigned int> &vertex_to_coinciding_vertex_group,
  const std::vector<std::set<typename Triangulation<dim>::active_cell_iterator>>
                                                      &v_to_c,
  typename DEM::dem_data_structures<dim>::cell_vector &periodic_neighbor_list)
{
  // Loop over vertices of the cell
  for (unsigned int vertex = 0; vertex < cell->n_vertices(); ++vertex)
    {
      // Get global id of vertex
      unsigned int vertex_id = cell->vertex_index(vertex);

      // Check if vertex is at periodic boundary, should be a key if so
      if (vertex_to_coinciding_vertex_group.find(vertex_id) !=
          vertex_to_coinciding_vertex_group.end())
        {
          // Get the coinciding vertex key to the current vertex
          unsigned int coinciding_vertex_key =
            vertex_to_coinciding_vertex_group.at(vertex_id);

          // Store the neighbor cells in list
          for (auto coinciding_vertex :
               coinciding_vertex_groups.at(coinciding_vertex_key))
            {
              // Skip the current vertex since we want only cells linked
              // to the periodic vertices
              if (coinciding_vertex != vertex_id)
                {
                  // Loop over all periodic neighbor
                  for (const auto &neighbor_id : v_to_c[coinciding_vertex])
                    {
                      periodic_neighbor_list.push_back(neighbor_id);
                    }
                }
            }
        }
    }
}

template class FindCellNeighbors<2>;
template class FindCellNeighbors<3>;

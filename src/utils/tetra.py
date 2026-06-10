import os
import torch

from typing import Tuple
from itertools import combinations


def parse_tet_file(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # Continue if file exists and its extensions is .tet, otherwise raise FileNotFound error
    if not os.path.isfile(path) or not path.endswith(".tet"):
        raise FileNotFoundError(path)

    with open(path, "r") as file: # Open .tet file
        template_verts = [] # Initialize list of at-rest vertices
        faces = [] # Initialize list of faces
        tetras = [] # Initialize list of tetrahedra

        while True: # Start parsing
            line: str = file.readline() # Read file line by line
            if not line: # Check if it is end of file, then break
                break

            words = line.strip("\n").split(" ") # Split line into list of words
            if words[0] == "v": # In case line containing vertice information
                template_verts.append([float(word) for word in words[1:-1]]) # Store vertice position
            elif words[0] == "t": # In case line containing tetrahedral information
                tetras.append(sorted([int(word) for word in words[1:]])) # Store tetrahedral information

        # Convert all lists to tensors
        template_verts = torch.tensor(template_verts).float() # Float type for position
        tetras = torch.tensor(tetras).long() # Long type for indexing

        # Eliminate duplicate tetrahedra
        tetras = tetras.unique(dim=-2)

        return template_verts, tetras

def extract_faces_from_tetras(
        tetras: torch.Tensor, # Tensor of tetrahedral vertice indices
        return_surface_faces: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    # Extract faces by permutation of tetrahedral vertice indices using tensor multiplication
    tetras_to_faces_tensor = torch.tensor([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
            [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ])

    # Extract all faces and sort
    faces = torch.matmul(tetras, tetras_to_faces_tensor)
    faces = faces.view(-1, 3) # Flatten face tensor into a matrix
    faces, _ = faces.sort(dim=-1) # Sort vertice indices in each face (for uniqueness filtering)

    if return_surface_faces: # If return surface faces
        faces, counts = faces.unique(dim=-2, return_counts=True) # Do uniqueness filtering and counting faces
        surface_faces = faces[counts == 1] # Surface faces having only one appearance
        return faces, surface_faces
    else: # If do not return surface faces
        faces = faces.unique(dim=-2)
        return faces, torch.empty(0, 3)

# Compute the relation (sparse) matrix between vertices and tetrahedra
# Entry (i, j) = 1 means that i-th vertice belongs to j-th tetrahedron
# Entry (i, j) = 0 means that i-th vertice does not belong to j-th tetrahedron
def compute_vert_to_tetra_relation(
        tetras: torch.Tensor, # Tensor of tetrahedral vertice indices
        return_n_tetras_per_vert: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    # This line is to turn on error message when sparse tensor operations crash
    torch.sparse.check_sparse_tensor_invariants.enable()

    i_indices = [] # List of first dimension indices
    j_indices = [] # List of second dimension indices
    for tetra, tetra_verts in enumerate(tetras):
        i_indices.extend([
                tetra_verts[0].item(),
                tetra_verts[1].item(),
                tetra_verts[2].item(),
                tetra_verts[3].item()
        ]) # Extend first list with indices of tetrahedral vertices
        j_indices.extend([tetra, tetra, tetra, tetra]) # Extend second list with tetrahedral indices
    indices = torch.tensor([i_indices, j_indices]) # Combine first and second dimension indices
    vert_to_tetra_relation_matrix = torch.sparse_coo_tensor(
        indices, torch.ones(indices.shape[-1])
    ).float()

    if return_n_tetras_per_vert: # If return number of tetrahedra corresponding to a vertice
        n_tetras_per_vert = torch.sparse.sum(
            vert_to_tetra_relation_matrix,
            dim=-1
        ).to_dense().float()
        return vert_to_tetra_relation_matrix, n_tetras_per_vert
    else: # If do not return number of tetrahedra corresponding to a vertice
        return vert_to_tetra_relation_matrix, torch.empty(0, 1)

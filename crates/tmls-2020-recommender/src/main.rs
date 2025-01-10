use anyhow::Result;

use nalgebra::DVector;
use tmls_2020_recommender::prelude::*;
#[tokio::main]
async fn main() -> Result<()> {
    let pool = create_test_pool(1);
    let mut conn = pool.get().unwrap();

    // Create the user-item matrix
    let (matrix, mapper) = create_user_item_matrix(&mut conn).unwrap();

    // Print matrix dimensions
    dbg!(matrix.ncols(), matrix.nrows());

    // Compute item-item similarity matrix
    let similarity_matrix = compute_item_item_similarity(&matrix);

    // Find the top-10 nearest neighbors for each item
    let neighbors = find_k_nearest_neighbors(&similarity_matrix, 10);
    let movie_index = mapper.get_movie_index(1961).unwrap_or_default();
    if let Some(sim) = neighbors.get(&movie_index) {
        for movie in sim {
            println!(
                "Movie ID: {:?}, Similarity: {}",
                mapper.get_movie_id_by_index(movie.0 as i64),
                movie.1
            );
        }
    }

    // Recommend items for user 0
    // let user_ratings = DVector::from_vec(matrix.row(1).iter().copied().collect());

    // let recommendations = recommend_items_for_user(&user_ratings, &neighbors);

    // println!("Recommendations for user {}: {:?}", 0, recommendations);

    Ok(())
}

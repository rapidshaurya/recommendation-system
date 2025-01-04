use anyhow::Context;
use diesel::*;
use ndarray::Array2;
#[derive(Debug, Queryable, Selectable)]
#[diesel(table_name = crate::schema::ratings)]
pub struct UserMovieRating {
    pub user_id: i64,
    pub movie_id: i64,
    pub rating: Option<f64>,
}

pub fn create_user_item_matrix(
    conn: &mut pg::PgConnection,
    num_users: usize,
    num_items: usize,
) -> anyhow::Result<Array2<f64>> {
    use crate::schema::ratings;
    let ratings: Vec<UserMovieRating> = ratings::table
        .select(UserMovieRating::as_select())
        .get_results(conn)?;
    let mut matrix = Array2::<f64>::zeros((num_users, num_items));
    let a = matrix.axes();

    for data in ratings {
        let user = data.user_id as usize;
        let item = data.movie_id as usize;
        let rating = data.rating.unwrap_or_default();
        if user < matrix.nrows() && item < matrix.ncols() {
            matrix[[user, item]] = rating;
        } else {
            eprintln!(
                "Skipping out-of-bounds entry: user={} item={} rating={}",
                user + 1,
                item + 1,
                rating // Adding 1 for human-readable output
            );
        }
    }

    Ok(matrix)
}

pub fn most_frequently_rated_films(ratings: &Array2<f64>, top_n: usize) {
    let mut film_rating_count: Vec<(usize, usize)> = Vec::new();

    // Iterate over columns (films) and count the number of ratings for each film
    for j in 0..ratings.ncols() {
        let mut count = 0;

        // Iterate over rows (users) for each film
        for i in 0..ratings.nrows() {
            let rating = ratings[[i, j]];
            if rating > 0.0 {
                // Consider only rated films (non-zero ratings)
                count += 1;
            }
        }

        // Store the count of ratings for the current film (column index)
        film_rating_count.push((j, count));
    }

    // Sort films by rating count in descending order
    film_rating_count.sort_by(|a, b| b.1.cmp(&a.1));

    // Print the top N most frequently rated films with their IDs
    for (film_id, count) in film_rating_count.into_iter().take(top_n) {
        println!("Film ID: {}, Number of Ratings: {}", film_id, count);
    }
}

pub fn highest_rated_films(ratings: &Array2<f64>, top_n: usize) -> Vec<(usize, f64)> {
    let mut film_ratings: Vec<(usize, f64)> = Vec::new();

    // Iterate over columns (films) and calculate the average rating for each film
    for j in 0..ratings.ncols() {
        let mut sum = 0.0;
        let mut count = 0;

        // Iterate over rows (users) for each film
        for i in 0..ratings.nrows() {
            let rating = ratings[[i, j]];
            if rating > 0.0 {
                // Consider only rated films
                sum += rating;
                count += 1;
            }
        }

        // Compute the average rating for the current film
        if count > 0 {
            let avg_rating = sum / count as f64;
            film_ratings.push((j, avg_rating));
        }
    }

    // Sort films by average rating in descending order
    film_ratings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Return the top N highest-rated films
    let data: Vec<(usize, f64)> = film_ratings.into_iter().take(top_n).collect();

    // Print the top N most frequently rated films with their IDs
    for (film_id, count) in &data {
        println!("Film ID: {}, Rating: {}", film_id, count);
    }
    data
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::prelude::create_test_pool;

    #[test]
    fn test_user_matrix() {
        let pool = create_test_pool(1);
        let mut conn = pool.get().unwrap();
        let matrix = create_user_item_matrix(&mut conn, 612, 200000).unwrap();
        println!("freq rated top 10 ");
        most_frequently_rated_films(&matrix, 10);
        println!("highest rating top 10 ");
        highest_rated_films(&matrix, 10);
    }
}

// @generated automatically by Diesel CLI.

diesel::table! {
    movies (movie_id) {
        movie_id -> Int8,
        title -> Nullable<Varchar>,
        genres -> Nullable<Varchar>,
    }
}

diesel::table! {
    ratings (user_id, movie_id) {
        user_id -> Int8,
        movie_id -> Int8,
        rating -> Nullable<Float8>,
        timestamp -> Nullable<Int8>,
    }
}

diesel::allow_tables_to_appear_in_same_query!(
    movies,
    ratings,
);

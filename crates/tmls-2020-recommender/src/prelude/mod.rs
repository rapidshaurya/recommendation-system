use diesel::{
    r2d2::{self, ConnectionManager},
    PgConnection,
};

pub mod user_rating_matrix;
pub use user_rating_matrix::*;
pub type DbPool = r2d2::Pool<ConnectionManager<PgConnection>>;

pub fn create_test_pool(size: u32) -> DbPool {
    use diesel::r2d2;
    use diesel::r2d2::ConnectionManager;
    use diesel::PgConnection;
    use dotenvy::dotenv;
    use std::env;

    dotenv().ok();
    let database_url: String = env::var("TEST_DATABASE_URL").unwrap();
    let manager = ConnectionManager::<PgConnection>::new(database_url);
    let pool = r2d2::Pool::builder().max_size(size).build(manager).unwrap();
    pool
}

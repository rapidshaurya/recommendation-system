use anyhow::Result;
use datafusion::arrow::array::{Float64Array, Int64Array};
use datafusion::prelude::*;
use ndarray::Array2;
pub async fn create_user_item_matrix(ctx: &SessionContext) -> Result<Array2<f64>> {
    // Fetch ratings from the table
    let query = r#"
        SELECT "userId", "movieId", rating
        FROM ratings
    "#;

    let df = ctx.sql(query).await?;
    let batches = df.clone().collect().await?;

    let schema = df.schema();
    println!("Schema: {:?}", schema);

    // Define user and item counts (for demo purposes; replace with dynamic detection if needed)
    let user_count = 1000; // Adjust based on your data
    let item_count = 500; // Adjust based on your data
    let mut matrix = Array2::<f64>::zeros((user_count, item_count));

    // Map column names to their indices
    let user_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "userId")
        .unwrap();
    let item_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "movieId")
        .unwrap();
    let rating_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "rating")
        .unwrap();

    for batch in batches {
        let user_col = batch
            .column(user_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let item_col = batch
            .column(item_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let rating_col = batch
            .column(rating_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        for i in 0..user_col.len() {
            let user = user_col.value(i) as usize - 1; // Subtract 1 if userId starts from 1
            let item = item_col.value(i) as usize - 1; // Subtract 1 if movieId starts from 1
            let rating = rating_col.value(i);

            // Ensure indices are within bounds
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
    }

    Ok(matrix)
}

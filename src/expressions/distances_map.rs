use datafusion::arrow::array::{
    ArrayRef, Int32Array, StructArray, as_primitive_array, as_struct_array,
};
use datafusion::arrow::compute::min;
use datafusion::arrow::datatypes::{DataType, Field, Fields, Int32Type};
use datafusion::common::ScalarValue;
use datafusion::error::Result;
use datafusion::physical_plan::Accumulator;
use std::collections::HashMap;
use std::sync::Arc;

/// Internal accumulator type for maintaining the shortest path distances from landmarks.
///
/// This accumulator keeps track of the minimum distances from each vertex to a set of landmark vertices.
/// The distances are stored in a HashMap where keys are landmark IDs and values are minimum distances.
#[derive(Debug)]
pub(crate) struct DistancesMap {
    /// Maps landmark vertex IDs to their current minimum distances
    distances: HashMap<i64, i32>,
}

impl DistancesMap {
    pub(crate) fn new(landmarks: Arc<Vec<i64>>) -> Self {
        Self {
            distances: HashMap::from_iter(landmarks.iter().map(|&lm| (lm, i32::MAX))),
        }
    }
}

/// Implementation of DataFusion's Accumulator trait for DistancesMap.
///
/// The accumulator maintains and updates minimum distances from vertices to landmarks:
/// 1. Update batch processes with new distance values:
///    - Examines each landmark's distances from incoming struct arrays
///    - Updates the stored distance if the incoming value is smaller
/// 2. Evaluate converts the current state to ScalarValue
/// 3. State converts distances to struct array format with:
///    - Fields for each landmark ID
///    - Arrays containing current minimum distances
/// 4. Merge batch combines multiple states using the same logic as update
///
impl Accumulator for DistancesMap {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let array_of_structs = as_struct_array(&values[0]);
        for key in self.distances.clone().keys() {
            let array_for_key = array_of_structs.column_by_name(&key.to_string());
            let min_distance_from_incoming = array_for_key
                .map(|array| min(as_primitive_array::<Int32Type>(array)))
                .unwrap_or(Some(i32::MAX))
                .unwrap_or(i32::MAX);

            if min_distance_from_incoming < self.distances[key] {
                self.distances.insert(*key, min_distance_from_incoming);
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        self.state().map(|state| state[0].clone())
    }

    fn size(&self) -> usize {
        size_of::<Self>()
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut sorted_keys = self.distances.keys().clone().collect::<Vec<_>>();
        sorted_keys.sort();
        let fields = Fields::from(
            sorted_keys
                .iter()
                .map(|k| Field::new(k.to_string(), DataType::Int32, true))
                .collect::<Vec<_>>(),
        );
        let arrays = sorted_keys
            .iter()
            .map(|k| self.distances.get(k).unwrap())
            .map(|v| Int32Array::from(vec![*v]))
            .map(|arr| Arc::new(arr) as ArrayRef)
            .collect::<Vec<_>>();
        let nulls = None;
        let scalar_value = ScalarValue::Struct(Arc::new(StructArray::new(fields, arrays, nulls)));

        Ok(vec![scalar_value])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.update_batch(states)
    }
}

#[derive(Copy, Clone, Debug, Default)]
/// For each cell of the matrix calculation an action is executed in order to align the X time
/// series to Y. This can be either of insertion, deletion or matching.
pub enum Action {
    /// Insertion - samples are inserted into the X time series in order to align it to Y
    Inserted,
    /// Deletion - samples are deleted from the X time series in order to align it to Y
    Deleted,
    /// Matching - samples from X are found to be aligned with samples for Y
    Matched,
    #[default]
    /// Default enum value used for .default() calls.
    Unknown,
}

#[derive(Copy, Clone)]
pub enum DistanceMode {
    Manhattan,
    Euclidean,
}

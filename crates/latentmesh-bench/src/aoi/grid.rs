//! `grid_v1`: independent full Cartesian product of the seven axes.

#[derive(Clone, Debug)]
pub struct Axes {
    pub age_ms: Vec<u64>,
    pub budget_bytes: Vec<u64>,
    pub queue_depth: Vec<u64>,
    pub loss_ppm: Vec<u64>,
    pub reorder_ppm: Vec<u64>,
    pub outage_ms: Vec<u64>,
    pub relevance_ppm: Vec<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub id: u64,
    pub age_ms: u64,
    pub budget_bytes: u64,
    pub queue_depth: u64,
    pub loss_ppm: u64,
    pub reorder_ppm: u64,
    pub outage_ms: u64,
    pub relevance_ppm: u64,
}

impl Axes {
    /// The registered 648-cell grid.
    pub fn full() -> Self {
        Self {
            age_ms: vec![0, 500, 2000],
            budget_bytes: vec![600, 1200, 2400],
            queue_depth: vec![4, 16],
            loss_ppm: vec![0, 100_000, 300_000],
            reorder_ppm: vec![0, 200_000],
            outage_ms: vec![0, 5000],
            relevance_ppm: vec![250_000, 500_000, 1_000_000],
        }
    }

    /// CI smoke subset (32 cells); never frozen.
    pub fn quick() -> Self {
        Self {
            age_ms: vec![0, 2000],
            budget_bytes: vec![600, 1200],
            queue_depth: vec![4],
            loss_ppm: vec![0, 300_000],
            reorder_ppm: vec![0],
            outage_ms: vec![0, 5000],
            relevance_ppm: vec![250_000, 1_000_000],
        }
    }

    /// Cells in axis-table order, first axis outermost, `cell_id` = index.
    pub fn cells(&self) -> Vec<Cell> {
        let mut out = Vec::new();
        for &age_ms in &self.age_ms {
            for &budget_bytes in &self.budget_bytes {
                for &queue_depth in &self.queue_depth {
                    for &loss_ppm in &self.loss_ppm {
                        for &reorder_ppm in &self.reorder_ppm {
                            for &outage_ms in &self.outage_ms {
                                for &relevance_ppm in &self.relevance_ppm {
                                    out.push(Cell {
                                        id: out.len() as u64,
                                        age_ms,
                                        budget_bytes,
                                        queue_depth,
                                        loss_ppm,
                                        reorder_ppm,
                                        outage_ms,
                                        relevance_ppm,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

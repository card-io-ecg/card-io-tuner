use std::ops::Index;

use crate::data::matrix::Matrix;

#[derive(Debug)]
pub struct GroupMap {
    /// for each cycle index, the group it belongs to
    map: Vec<usize>,

    /// for each group, the number of cycles in that group
    counts: Vec<usize>,
}

impl GroupMap {
    pub fn iter(&self) -> impl Iterator<Item = Group<'_>> {
        (0..self.counts.len()).map(move |group| Group {
            map: self,
            index: group,
        })
    }

    pub fn similar_count(&self, cycle: usize) -> usize {
        self.counts.get(self.map[cycle]).copied().unwrap_or(0)
    }

    pub fn group(&self, group_idx: usize) -> Group<'_> {
        Group {
            map: self,
            index: group_idx,
        }
    }
}

impl Index<usize> for GroupMap {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.map[index]
    }
}

pub struct Group<'a> {
    map: &'a GroupMap,
    index: usize,
}

impl Group<'_> {
    pub fn cycles(&self) -> impl Iterator<Item = usize> + Clone + '_ {
        self.map
            .map
            .iter()
            .enumerate()
            .filter(|(_, group)| **group == self.index)
            .map(|(cycle, _)| cycle)
    }

    pub fn len(&self) -> usize {
        self.map.counts[self.index]
    }

    pub fn index(&self) -> usize {
        self.index
    }
}

pub fn group_cycles(data: &Matrix<f32>, threshold: f32) -> GroupMap {
    let mut group_map: Vec<usize> = vec![usize::MAX; data.rows()];
    let mut group_counts: Vec<usize> = vec![];
    let mut group_count = 0;

    for cycle_idx in 0..data.rows() {
        let (_, target_group) = (0..group_count).fold(
            (f32::MAX, usize::MAX),
            |(min_distance, target_group), current_group| {
                let mut cycles_in_group = 0;
                let mut sum = 0.0;

                for (other_cycle, other_group) in group_map.iter().take(cycle_idx).enumerate() {
                    if *other_group == current_group {
                        cycles_in_group += 1;
                        sum += data[(cycle_idx, other_cycle)];
                    }
                }

                let avg = sum / cycles_in_group as f32;

                if avg >= threshold && avg < min_distance {
                    (avg, current_group)
                } else {
                    (min_distance, target_group)
                }
            },
        );

        if target_group == usize::MAX {
            // create a new group
            group_map[cycle_idx] = group_count;
            group_count += 1;
            group_counts.push(1);
        } else {
            group_counts[target_group] += 1;
            group_map[cycle_idx] = target_group;
        }
    }

    log::debug!("group map: {:?}", group_map);
    log::debug!("group counts: {:?}", group_counts);

    GroupMap {
        map: group_map,
        counts: group_counts,
    }
}

#[cfg(test)]
mod test {
    use crate::data::{grouping::group_cycles, matrix::Matrix};

    #[test]
    fn test() {
        #[rustfmt::skip]
        let values: [[f32; 22]; 22] = [
            [1.0, 0.972, 0.983, 0.987, 0.856, 0.967, 0.959, 0.811, 0.786, 0.955, 0.955, 0.967, 0.969, 0.979, 0.973, 0.980, 0.977, 0.712, 0.973, 0.965, 0.962, 0.979],
            [0.972, 1.0, 0.985, 0.972, 0.901, 0.984, 0.971, 0.751, 0.687, 0.953, 0.970, 0.980, 0.983, 0.970, 0.968, 0.965, 0.980, 0.674, 0.981, 0.979, 0.974, 0.979],
            [0.983, 0.985, 1.0, 0.984, 0.881, 0.975, 0.969, 0.802, 0.725, 0.966, 0.972, 0.974, 0.984, 0.983, 0.976, 0.979, 0.985, 0.673, 0.987, 0.979, 0.975, 0.985],
            [0.987, 0.972, 0.984, 1.0, 0.876, 0.961, 0.970, 0.824, 0.775, 0.964, 0.966, 0.973, 0.977, 0.984, 0.984, 0.985, 0.977, 0.726, 0.980, 0.973, 0.973, 0.984],
            [0.856, 0.901, 0.881, 0.876, 1.0, 0.898, 0.915, 0.615, 0.532, 0.849, 0.876, 0.911, 0.910, 0.867, 0.887, 0.865, 0.880, 0.725, 0.896, 0.911, 0.909, 0.892],
            [0.967, 0.984, 0.975, 0.961, 0.898, 1.0, 0.957, 0.707, 0.688, 0.925, 0.950, 0.970, 0.970, 0.950, 0.949, 0.948, 0.971, 0.675, 0.965, 0.967, 0.958, 0.963],
            [0.959, 0.971, 0.969, 0.970, 0.915, 0.957, 1.0, 0.750, 0.666, 0.948, 0.965, 0.978, 0.976, 0.969, 0.974, 0.952, 0.968, 0.710, 0.971, 0.979, 0.989, 0.980],
            [0.811, 0.751, 0.802, 0.824, 0.615, 0.707, 0.750, 1.0, 0.753, 0.849, 0.797, 0.730, 0.761, 0.827, 0.831, 0.852, 0.786, 0.527, 0.816, 0.774, 0.769, 0.799],
            [0.786, 0.687, 0.725, 0.775, 0.532, 0.688, 0.666, 0.753, 1.0, 0.691, 0.650, 0.696, 0.691, 0.743, 0.749, 0.793, 0.720, 0.739, 0.711, 0.670, 0.669, 0.722],
            [0.955, 0.953, 0.966, 0.964, 0.849, 0.925, 0.948, 0.849, 0.691, 1.0, 0.982, 0.939, 0.953, 0.969, 0.970, 0.963, 0.957, 0.587, 0.975, 0.969, 0.959, 0.970],
            [0.955, 0.970, 0.972, 0.966, 0.876, 0.950, 0.965, 0.797, 0.650, 0.982, 1.0, 0.956, 0.962, 0.963, 0.970, 0.957, 0.966, 0.591, 0.977, 0.976, 0.971, 0.970],
            [0.967, 0.980, 0.974, 0.973, 0.911, 0.970, 0.978, 0.730, 0.696, 0.939, 0.956, 1.0, 0.985, 0.971, 0.970, 0.962, 0.980, 0.727, 0.966, 0.975, 0.977, 0.980],
            [0.969, 0.983, 0.984, 0.977, 0.910, 0.970, 0.976, 0.761, 0.691, 0.953, 0.962, 0.985, 1.0, 0.978, 0.973, 0.972, 0.983, 0.702, 0.977, 0.978, 0.980, 0.985],
            [0.979, 0.970, 0.983, 0.984, 0.867, 0.950, 0.969, 0.827, 0.743, 0.969, 0.963, 0.971, 0.978, 1.0, 0.981, 0.982, 0.981, 0.695, 0.980, 0.974, 0.978, 0.987],
            [0.973, 0.968, 0.976, 0.984, 0.887, 0.949, 0.974, 0.831, 0.749, 0.970, 0.970, 0.970, 0.973, 0.981, 1.0, 0.982, 0.974, 0.714, 0.978, 0.973, 0.977, 0.984],
            [0.980, 0.965, 0.979, 0.985, 0.865, 0.948, 0.952, 0.852, 0.793, 0.963, 0.957, 0.962, 0.972, 0.982, 0.982, 1.0, 0.976, 0.722, 0.976, 0.964, 0.960, 0.979],
            [0.977, 0.980, 0.985, 0.977, 0.880, 0.971, 0.968, 0.786, 0.720, 0.957, 0.966, 0.980, 0.983, 0.981, 0.974, 0.976, 1.0, 0.685, 0.975, 0.974, 0.973, 0.980],
            [0.712, 0.674, 0.673, 0.726, 0.725, 0.675, 0.710, 0.527, 0.739, 0.587, 0.591, 0.727, 0.702, 0.695, 0.714, 0.722, 0.685, 1.0, 0.660, 0.657, 0.688, 0.696],
            [0.973, 0.981, 0.987, 0.980, 0.896, 0.965, 0.971, 0.816, 0.711, 0.975, 0.977, 0.966, 0.977, 0.980, 0.978, 0.976, 0.975, 0.660, 1.0, 0.987, 0.977, 0.985],
            [0.965, 0.979, 0.979, 0.973, 0.911, 0.967, 0.979, 0.774, 0.670, 0.969, 0.976, 0.975, 0.978, 0.974, 0.973, 0.964, 0.974, 0.657, 0.987, 1.0, 0.985, 0.984],
            [0.962, 0.974, 0.975, 0.973, 0.909, 0.958, 0.989, 0.769, 0.669, 0.959, 0.971, 0.977, 0.980, 0.978, 0.977, 0.960, 0.973, 0.688, 0.977, 0.985, 1.0, 0.986],
            [0.979, 0.979, 0.985, 0.984, 0.892, 0.963, 0.980, 0.799, 0.722, 0.970, 0.970, 0.980, 0.985, 0.987, 0.984, 0.979, 0.980, 0.696, 0.985, 0.984, 0.986, 1.0],
        ];

        let matrix = Matrix::from(values);

        let result = group_cycles(&matrix, 0.85);

        assert_eq!(
            result.map,
            [0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 1, 3, 1, 1, 1, 1]
        );
        assert_eq!(result.counts, [13, 7, 1, 1]);
    }

    #[test]
    fn test2() {
        #[rustfmt::skip]
        let values: [[f32; 18]; 18] = [
            [1.0, 0.445, 0.817, 0.565, 0.828, 0.859, 0.810, 0.816, 0.819, 0.859, 0.473, 0.773, 0.761, 0.824, 0.806, 0.820, 0.820, -0.158],
            [0.445, 1.0, 0.400, 0.167, 0.385, 0.434, 0.425, 0.433, 0.390, 0.436, 0.602, 0.293, 0.398, 0.449, 0.418, 0.423, 0.437, 0.102],
            [0.817, 0.400, 1.0, 0.891, 0.991, 0.976, 0.975, 0.987, 0.990, 0.978, 0.484, 0.964, 0.985, 0.981, 0.987, 0.985, 0.988, -0.008],
            [0.565, 0.167, 0.891, 1.0, 0.878, 0.865, 0.896, 0.885, 0.895, 0.853, 0.338, 0.940, 0.911, 0.839, 0.903, 0.886, 0.885, -0.031],
            [0.828, 0.385, 0.991, 0.878, 1.0, 0.974, 0.964, 0.981, 0.987, 0.983, 0.470, 0.961, 0.981, 0.985, 0.983, 0.983, 0.983, -0.005],
            [0.859, 0.434, 0.976, 0.865, 0.974, 1.0, 0.982, 0.988, 0.987, 0.980, 0.522, 0.963, 0.970, 0.972, 0.988, 0.988, 0.986, -0.071],
            [0.810, 0.425, 0.975, 0.896, 0.964, 0.982, 1.0, 0.988, 0.985, 0.974, 0.489, 0.972, 0.977, 0.965, 0.988, 0.981, 0.987, -0.085],
            [0.816, 0.433, 0.987, 0.885, 0.981, 0.988, 0.988, 1.0, 0.993, 0.982, 0.501, 0.966, 0.989, 0.984, 0.993, 0.992, 0.992, -0.030],
            [0.819, 0.390, 0.990, 0.895, 0.987, 0.987, 0.985, 0.993, 1.0, 0.985, 0.473, 0.974, 0.989, 0.984, 0.992, 0.990, 0.990, -0.057],
            [0.859, 0.436, 0.978, 0.853, 0.983, 0.980, 0.974, 0.982, 0.985, 1.0, 0.476, 0.951, 0.977, 0.984, 0.978, 0.987, 0.976, -0.096],
            [0.473, 0.602, 0.484, 0.338, 0.470, 0.522, 0.489, 0.501, 0.473, 0.476, 1.0, 0.405, 0.468, 0.496, 0.502, 0.498, 0.506, 0.143],
            [0.773, 0.293, 0.964, 0.940, 0.961, 0.963, 0.972, 0.966, 0.974, 0.951, 0.405, 1.0, 0.965, 0.940, 0.973, 0.960, 0.966, -0.088],
            [0.761, 0.398, 0.985, 0.911, 0.981, 0.970, 0.977, 0.989, 0.989, 0.977, 0.468, 0.965, 1.0, 0.983, 0.987, 0.986, 0.983, -0.015],
            [0.824, 0.449, 0.981, 0.839, 0.985, 0.972, 0.965, 0.984, 0.984, 0.984, 0.496, 0.940, 0.983, 1.0, 0.976, 0.980, 0.979, -0.024],
            [0.806, 0.418, 0.987, 0.903, 0.983, 0.988, 0.988, 0.993, 0.992, 0.978, 0.502, 0.973, 0.987, 0.976, 1.0, 0.991, 0.992, -0.019],
            [0.820, 0.423, 0.985, 0.886, 0.983, 0.988, 0.981, 0.992, 0.990, 0.987, 0.498, 0.960, 0.986, 0.980, 0.991, 1.0, 0.986, -0.040],
            [0.820, 0.437, 0.988, 0.885, 0.983, 0.986, 0.987, 0.992, 0.990, 0.976, 0.506, 0.966, 0.983, 0.979, 0.992, 0.986, 1.0, -0.005],
            [-0.158, 0.102, -0.008, -0.031, -0.005, -0.071, -0.085, -0.030, -0.057, -0.096, 0.143, -0.088, -0.015, -0.024, -0.019, -0.040, -0.005, 1.0],
        ];

        let matrix = Matrix::from(values);

        let result = group_cycles(&matrix, 0.85);

        assert_eq!(
            result.map,
            [0, 1, 2, 2, 2, 0, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 2, 4]
        );
        assert_eq!(result.counts, [9, 1, 6, 1, 1]);
    }
}

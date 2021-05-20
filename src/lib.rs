use rand::thread_rng;
use rand::distributions::{Distribution, Uniform, WeightedIndex};

#[allow(dead_code)] // distance is only used in the test code, for now, as it is used strictly as a parameter during initialization.
pub struct Kmeans<T, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V> {
    means: Vec<T>,
    distance: D
}

impl <T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd + Into<f64>, D: Fn(&T,&T) -> V> Kmeans<T,V,D> {
    pub fn new<M: Fn(&Vec<&T>) -> T>(k: usize, data: &[T], distance: D, mean: M) -> Kmeans<T,V,D> {
        Kmeans {means: kmeans_iterate(k, data, &distance, &mean), distance}
    }

    #[cfg(test)]
    pub fn k(&self) -> usize {self.means.len()}

    #[cfg(test)]
    pub fn classification(&self, sample: &T) -> usize {
        classify(sample, &self.means, &self.distance)
    }

    #[cfg(test)]
    pub fn copy_means(&self) -> Vec<T> {self.means.clone()}

    pub fn move_means(self) -> Vec<T> {self.means}
}

pub fn initial_plus_plus<T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd + Into<f64>, D: Fn(&T,&T) -> V>
(k: usize, distance: &D, data: &[T]) -> Vec<T> {
    let mut result = Vec::new();
    let mut candidates: Vec<T> = data.iter().map(|t| t.clone()).collect();
    let range = Uniform::new(0, candidates.len());
    let mut rng = thread_rng();
    while candidates.len() < k {
        candidates.push(data[range.sample(&mut rng)].clone());
    }
    result.push(remove_random(&mut candidates, range));
    while result.len() < k {
        let squared_distances: Vec<f64> = candidates.iter()
            .map(|datum| 1.0f64 + distance(datum, result.last().unwrap()).into())
            .map(|dist| dist.powf(2.0))
            .collect();
        let dist = WeightedIndex::new(&squared_distances).unwrap();
        result.push(remove_random(&mut candidates, dist));
    }
    result
}

pub fn remove_random<T: Clone, D: Distribution<usize>>(candidates: &mut Vec<T>, distribution: D) -> T {
    let mut rng = thread_rng();
    let end = candidates.len() - 1;
    let choice = distribution.sample(&mut rng);
    candidates.swap(end, choice);
    candidates.remove(end)
}

fn kmeans_iterate<T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd + Into<f64>, D: Fn(&T,&T) -> V, M: Fn(&Vec<&T>) -> T>
(k: usize, data: &[T], distance: &D, mean: &M) -> Vec<T> {
    let mut result = initial_plus_plus(k, distance, data);
    loop {
        let mut classifications: Vec<Vec<&T>> = (0..k).map(|_| Vec::new()).collect();
        for datum in data {
            let category = classify(datum, &result, distance);
            classifications[category].push(datum);
        }
        let prev = result;
        result = classifications.iter().enumerate()
            .map(|(i, c)|
                if c.is_empty() {
                    prev[i].clone()
                } else {
                    mean(c)
                }
            ).collect();

        if result.iter().enumerate().all(|(i,r)| &prev[i] == r) {
            return result;
        }
    }
}

fn classify<T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V>(target: &T, means: &Vec<T>, distance: &D) -> usize {
    let distances: Vec<(V,usize)> = (0..means.len())
        .map(|i| (distance(&target, &means[i]).into(), i))
        .collect();
    distances.iter()
        .fold(None, |m:Option<&(V, usize)>, d| m.map_or(Some(d), |m|
            Some(if m.0 < d.0 {m} else {d}))).unwrap().1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manhattan(n1: &i32, n2: &i32) -> i32 {
        let mut diff = n1 - n2;
        if diff < 0 {diff = -diff;}
        diff
    }

    fn mean(nums: &Vec<&i32>) -> i32 {
        let total: i32 = nums.iter().map(|i| *i).sum();
        total / (nums.len() as i32)
    }

    #[test]
    fn test_k_means() {
        let target_means = vec![3, 11, 25, 40];
        let num_target_means = target_means.len();
        let data = vec![2, 3, 4, 10, 11, 12, 24, 25, 26, 35, 40, 45];
        let kmeans = Kmeans::new(num_target_means, &data, manhattan, mean);
        let mut sorted_means = kmeans.copy_means();
        sorted_means.sort();
        assert_eq!(kmeans.k(), sorted_means.len());
        assert_eq!(sorted_means.len(), num_target_means);
        assert_eq!(sorted_means, target_means);
    }

    #[test]
    fn test_underflow() {
        let data = vec![1, 2, 3];
        let k = data.len() + 1;
        let kmeans = Kmeans::new(k, &data, manhattan, mean);
        assert_eq!(kmeans.means.len(), k);
        for datum in data.iter() {
            assert!(kmeans.means.contains(datum));
        }
    }
}
use rand::thread_rng;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use std::sync::Arc;

#[derive(Clone)]
pub struct Kmeans<T, V: Copy + PartialEq + PartialOrd, D: Fn(&T,&T) -> V> {
    means: Vec<T>,
    distance: Arc<D>
}

impl <T: Clone + PartialEq, V: Copy + PartialEq + PartialOrd + Into<f64>, D: Fn(&T,&T) -> V> Kmeans<T,V,D> {
    pub fn new<M: Fn(&Vec<&T>) -> T>(k: usize, data: &[T], distance: Arc<D>, mean: Arc<M>) -> Kmeans<T,V,D> {
        Kmeans {means: Self::kmeans_iterate(k, data, &distance, mean), distance}
    }

    pub fn k(&self) -> usize {self.means.len()}

    pub fn classification(&self, sample: &T) -> usize {
        Self::classify(sample, &self.means, &self.distance)
    }

    pub fn best_matching_mean(&self, sample: &T) -> T {
        self.means[self.classification(sample)].clone()
    }

    pub fn copy_means(&self) -> Vec<T> {self.means.clone()}

    pub fn move_means(self) -> Vec<T> {self.means}

    pub fn initial_plus_plus(k: usize, distance: &D, data: &[T]) -> Vec<T> {
        let mut result = Vec::new();
        let mut candidates: Vec<T> = data.iter().map(|t| t.clone()).collect();
        let range = Uniform::new(0, candidates.len());
        let mut rng = thread_rng();
        while candidates.len() < k {
            candidates.push(data[range.sample(&mut rng)].clone());
        }
        result.push(Self::remove_random(&mut candidates, range));
        while result.len() < k {
            let squared_distances: Vec<f64> = candidates.iter()
                .map(|datum| 1.0f64 + distance(datum, result.last().unwrap()).into())
                .map(|dist| dist.powf(2.0))
                .collect();
            let dist = WeightedIndex::new(&squared_distances).unwrap();
            result.push(Self::remove_random(&mut candidates, dist));
        }
        result
    }

    pub fn remove_random<P: Distribution<usize>>(candidates: &mut Vec<T>, distribution: P) -> T {
        let mut rng = thread_rng();
        let end = candidates.len() - 1;
        let choice = distribution.sample(&mut rng);
        candidates.swap(end, choice);
        candidates.remove(end)
    }

    fn kmeans_iterate<M: Fn(&Vec<&T>) -> T>(k: usize, data: &[T], distance: &D, mean: Arc<M>) -> Vec<T> {
        let mut result = Self::initial_plus_plus(k, distance, data);
        loop {
            let mut classifications: Vec<Vec<&T>> = (0..k).map(|_| Vec::new()).collect();
            for datum in data {
                let category = Self::classify(datum, &result, distance);
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

    fn classify(target: &T, means: &Vec<T>, distance: &D) -> usize {
        let distances: Vec<(V,usize)> = (0..means.len())
            .map(|i| (V::from(distance(&target, &means[i])), i))
            .collect();
        distances.iter()
            .fold(None, |m:Option<&(V, usize)>, d| m.map_or(Some(d), |m|
                Some(if m.0 < d.0 {m} else {d}))).unwrap().1
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

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
        let kmeans = Kmeans::new(num_target_means, &data, Arc::new(manhattan), Arc::new(mean));
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
        let kmeans = Kmeans::new(k, &data, Arc::new(manhattan), Arc::new(mean));
        assert_eq!(kmeans.means.len(), k);
        for datum in data.iter() {
            assert!(kmeans.means.contains(datum));
        }
    }

    #[test]
    fn test_mutex() {
        let shared = Arc::new(Mutex::new(None));
        {
            let shared = shared.clone();
            std::thread::spawn(move || {
                let values = (0..10).collect::<Vec<_>>();
                let kmeans = Kmeans::new(2, &values, Arc::new(manhattan), Arc::new(mean));
                let mut shared = shared.lock().unwrap();
                *shared = Some(kmeans);
            });
        }
        let mut quit = false;
        while !quit {
            let shared = shared.lock().unwrap();
            shared.as_ref().map(|shared| {
                println!("Completed; means: {:?}", shared.copy_means());
                quit = true;
            });
        }
    }
}

#[cfg(test)]
mod color_tests {
    use std::time::Instant;

    use super::*;
    use scarlet::prelude::{Color, RGBColor, ColorPoint};

    #[test]
    fn test_colors() {
        const WIDTH: usize = 64;
        const HEIGHT: usize = 48;
        const NUM_COLOR_CLUSTERS: usize = 2;

        let start = Instant::now();
        let image = (0..WIDTH*HEIGHT*4).map(|_| RGBColor {r: rand::random(), g: rand::random(), b: rand::random()}).collect::<Vec<_>>();
        let kmeans = Kmeans::new(NUM_COLOR_CLUSTERS, &image, Arc::new(RGBColor::distance), Arc::new(|items: &Vec<&RGBColor>| {
            let item = *items[0];
            let others = items[1..].iter().copied().copied().collect::<Vec<_>>();
            item.average(others).into()
        }));
        assert_eq!(kmeans.k(), NUM_COLOR_CLUSTERS);
        println!("elapsed time: {}s", start.elapsed().as_secs_f64());
    }
}
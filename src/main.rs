#![allow(dead_code)]
mod activation;

use crate::activation::Sigmoid;
use thiserror::Error;
use std::{cell::RefCell, fmt, rc::Rc};
use rand::Rng;

#[derive(Default, Clone)]
struct Connection {
    node: Rc<RefCell<Node>>,
    weight: f32,
}

impl Connection {
    fn new(node: Rc<RefCell<Node>>) -> Connection {
        let mut rng = rand::thread_rng();
        Connection {
            node: node,
            weight: rng.gen_range(-1.0..1.0),
        }
    }
}

impl fmt::Debug for Connection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Connection")
            .field("weight", &self.weight)
            .finish()
    }
}

#[derive(Default, Clone, Debug)]
struct Node {
    activation: f32,
    bias: f32,
    weighted_sum: f32,
    connections: Vec<Connection>
}

impl Node {
    fn new() -> Node {
        let mut rng = rand::thread_rng();
        Node {
            activation: 0.0,
            bias: rng.gen_range(-1.0..1.0),
            connections: vec![],
            weighted_sum: 0.0,
        }
    }

    fn add_connection(&mut self, node: Rc<RefCell<Node>>) {
        self.connections.push(Connection::new(node));
    }

    fn calculate_weighted_sum(&mut self) {
        self.weighted_sum = self.connections.iter().fold(0.0, |acc, x| acc + x.weight * x.node.borrow().activation) + self.bias;
    }


}

#[derive(Debug)]
struct Perceptron {
    input: Vec<Rc<RefCell<Node>>>,
    hidden_layers: Vec<Vec<Rc<RefCell<Node>>>>,
    output:Vec<Rc<RefCell<Node>>>
}

#[derive(Error, Debug)]
enum PerceptronError {
    #[error("Invalid input")]
    InvalidInputSize,
    #[error("Invalid configuration")]
    InvalidConfiguration,
}

impl Perceptron {
    fn calculate_weighted_sum(&mut self) {
        for i in self.hidden_layers.iter_mut() {
            for j in i.iter_mut() {
                j.borrow_mut().calculate_weighted_sum();
            }
        }

        for i in self.output.iter_mut() {
            i.borrow_mut().calculate_weighted_sum();
        }
    }

    fn new(input_num: i32, hidden_layers_num: i32, hidden_nodes_num: i32, output_num: i32) -> Result<Perceptron, PerceptronError> {
        if input_num == 0 || hidden_layers_num == 0 || hidden_nodes_num == 0 || output_num == 0 {
            return Err(PerceptronError::InvalidConfiguration);
        }


        let input_nodes = vec![Rc::new(RefCell::new(Node::new())); input_num as usize];
        let mut hidden_layers = vec![vec![Rc::new(RefCell::new(Node::new())); hidden_nodes_num as usize]; hidden_layers_num as usize];
        let mut output_nodes = vec![Rc::new(RefCell::new(Node::new())); output_num as usize];

        for i in hidden_layers[0].iter_mut() {
            for j in input_nodes.iter() {
                i.borrow_mut().add_connection(j.clone());
            }
        }

        for i in hidden_layers.len() - 1..0 {
            let (left, right) = hidden_layers.split_at_mut(i - 1);
            Perceptron::connect_layers(&mut right[i], &left.last().unwrap());
        }

        Perceptron::connect_layers(&mut output_nodes , &hidden_layers.last_mut().unwrap());
        let mut perceptron = Perceptron {
            input: input_nodes,
            hidden_layers: hidden_layers,
            output: output_nodes
        };

        perceptron.calculate_weighted_sum();

        Ok(perceptron)
    }

    fn connect_layers(layer_1: &mut Vec<Rc<RefCell<Node>>>, layer_2: &Vec<Rc<RefCell<Node>>>){
        for i in layer_1.iter_mut() {
            for j in layer_2.iter() {
                i.borrow_mut().add_connection(j.clone());
            }
        }
    }
    
}

fn main() {
    let perceptron = Perceptron::new(8, 2, 2, 2).unwrap();
    println!("{:?}", perceptron);
}

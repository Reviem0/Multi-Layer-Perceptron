use std::{cell::RefCell, rc::Rc};

#[derive(Default, Clone, Debug)]
struct Connection {
    node: Rc<RefCell<Node>>,
    weight: i32,
    bias: i32
}

impl Connection {
    fn new(node: Rc<RefCell<Node>>, weight: i32, bias: i32) -> Connection {
        Connection {
            node: node,
            weight: weight,
            bias: bias
        }
    }
}

#[derive(Default, Clone, Debug)]
struct Node {
    connections: Vec<Connection>
}

impl Node {
    fn new() -> Node {
        Node {
            connections: vec![]
        }
    }

    fn add_connection(&mut self, node: Rc<RefCell<Node>>) {
        self.connections.push(Connection::new(node, 0, 0));
    }


}

#[derive(Debug)]
struct Perceptron {
    input: Vec<Node>,
    hidden_layers: Vec<Vec<Rc<RefCell<Node>>>>,
    output:Vec<Node>
}

impl Perceptron {
    fn connect_layers(layer_1: &mut Vec<Rc<RefCell<Node>>>, layer_2: &Vec<Rc<RefCell<Node>>>){
        for i in layer_1.iter_mut() {
            for j in layer_2.iter() {
                i.borrow_mut().add_connection(j.clone());
            }
        }
    }

    fn new(input_num: i32, hidden_layers_num: i32, hidden_nodes_num: i32, output_num: i32) -> Perceptron {
        let mut input_nodes = vec![Node::default(); input_num as usize];
        let mut hidden_layers = vec![vec![Rc::new(RefCell::new(Node::default())); hidden_nodes_num as usize]; hidden_layers_num as usize];
        let output_nodes = vec![Node::default(); output_num as usize];

        for i in input_nodes.iter_mut() {
            for j in hidden_layers[0].iter() {
                i.add_connection(j.clone());
            }
        }

        for i in 0..hidden_layers.len() - 1 {
            Perceptron::connect_layers(&mut hidden_layers[i], &hidden_layers[i + 1]);
        }

        Perceptron::connect_layers(&mut hidden_layers.last().unwrap() , &output_nodes);

        Perceptron {
            input: input_nodes,
            hidden_layers: hidden_layers,
            output: output_nodes

        }
    }
    
}

fn main() {
    let perceptron = Perceptron::new(2, 2, 2, 1);
    println!("{:?}", perceptron);
}

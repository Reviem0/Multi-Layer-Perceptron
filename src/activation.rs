pub trait Activation {
    fn activate(x: f32) -> f32;
    fn derivative(x: f32) -> f32;
}

pub struct Sigmoid;
pub struct ReLU;

impl Activation for Sigmoid {
    fn activate(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derivative(x: f32) -> f32 {
        let sigmod = Sigmoid::activate(x);
        sigmod * (1.0 - sigmod)
    }
}


impl Activation for ReLU {
    fn activate(x: f32) -> f32 {
       x.max(0.0)
    }

    fn derivative(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}
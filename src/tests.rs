use minimizer::Minimizer;


#[test]
fn minimize_square() {
    let minimizer = Minimizer::default();
    let expected = 0.0;
    let result = minimizer
        .minimize(|x| x[0].powi(2), vec![1.0_f64])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-8);
}


#[test]
fn minimize_sin() {
    let minimizer = Minimizer::default();
    let expected = -1.0;
    let result = minimizer
        .minimize(|x| x[0].sin(), vec![0.0_f64])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-16);
}


#[test]
fn minimize_cosh() {
    let minimizer = Minimizer::default();
    let expected = 1.0;
    let result = minimizer
        .minimize(|x| x[0].cosh(), vec![1.0_f64])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-8);
}


#[test]
fn minimize_abs() {
    let minimizer = Minimizer::default();
    let expected = 0.0;
    let result = minimizer
        .minimize(|x| x[0].abs(), vec![1.0_f64])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-8);
}

use minimizer::Minimizer;


#[test]
fn minimize_square_1d() {
    let minimizer = Minimizer::default();
    let expected = 0.0;
    let result = minimizer
        .minimize(|x| x[0].powi(2), &vec![1.0_f64])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-9);
}


#[test]
fn minimize_square_2d() {
    let minimizer = Minimizer::default();
    let expected = 0.0;
    let result = minimizer
        .minimize(|x| x[0].powi(2) * x[1].powi(2), &vec![1.0_f64, 1.0])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-9);
}


#[test]
fn minimize_sin() {
    let minimizer = Minimizer::default();
    let expected = -1.0;
    let result = minimizer
        .minimize(|x| x[0].sin(), &vec![0.0_f64])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-9);
}


#[test]
fn minimize_cosh() {
    let minimizer = Minimizer::default();
    let expected = 1.0;
    let result = minimizer
        .minimize(|x| x[0].cosh(), &vec![1.0_f64])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-9);
}


#[test]
fn minimize_abs() {
    let minimizer = Minimizer::default();
    let expected = 0.0;
    let result = minimizer
        .minimize(|x| x[0].abs(), &vec![1.0_f64])
        .unwrap();

    println!("f_min = {:?}", result.f_min);
    println!("x_min = {:?}", result.x_min);
    println!(" iter = {:?}", result.iter);

    assert!((result.f_min - expected).abs() < 1e-9);
}

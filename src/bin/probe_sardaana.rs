//! -------------------------------------------------------------------------
//! Filename:    src/bin/poc_sardaana.rs
//! Class:       SemantikaEngine
//! Purpose:     poc-sardaana — существо с мозгом от трансформера
//! Description: грузим SarDNA_Pro_best.taas (256 нейронов, 13K синапсов,
//!              обучено на реальных коннектомах 14 видов) как кору.
//!              Смотрим топологию + гоняем жизнь.
//! author: Atlasov anton
// -------------------------------------------------------------------------
use taas_format::TaasReader;

fn main() {
    let path = "D:/Проекты/asm_engines/sardaanaCM/data/generated/SarDNA_Pro_best.taas";
    println!("poc-sardaana: грузим трансформерный коннектом");
    let r = TaasReader::from_file(path).expect("load taas");

    // Извлекаем CSR компоненты.
    let rp: Vec<i32> = r.get_i32("row_ptr").expect("row_ptr");
    let ci: Vec<i32> = r.get_i32("col_indices").expect("col_indices");
    let vals: Vec<f32> = r.get_f32("values").expect("values");
    let nt: Vec<f32> = r.get_f32("nt_types").expect("nt_types");

    let n_neurons = rp.len().saturating_sub(1);
    let n_syn = ci.len();
    let n_exc = nt.iter().filter(|&&x| x > 0.0).count();
    let n_inh = nt.iter().filter(|&&x| x < 0.0).count();

    println!(
        "SarDNA: {} нейронов · {} синапсов · E/I={}/{} ({:.0}% возб.)",
        n_neurons, n_syn, n_exc, n_inh,
        n_exc as f64 / (n_exc + n_inh).max(1) as f64 * 100.0
    );

    // Статистика весов.
    let w_min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
    let w_max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let w_mean = vals.iter().sum::<f32>() / vals.len() as f32;
    println!("веса: min={w_min:.3} max={w_max:.3} mean={w_mean:.3}");

    // Степени.
    let degrees: Vec<usize> = rp.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
    let deg_max = degrees.iter().max().unwrap_or(&0);
    let deg_mean: f64 = degrees.iter().map(|&d| d as f64).sum::<f64>() / n_neurons as f64;
    println!("степени: max={deg_max} mean={deg_mean:.1}");

    // Плотность.
    let density = n_syn as f64 / (n_neurons * (n_neurons - 1)) as f64;
    println!("плотность: {:.4}", density);

    println!("\nГОТОВО К ЗАГРУЗКЕ В DIOMA — те же поля что у ConnectomeData!");
}

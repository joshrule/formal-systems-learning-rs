//! [Rust][1] simulations using [2AFC][2] triads to learn [typed][3] first-order
//! [term rewriting systems][4] modeling various versions of the [MU formal
//! system][5].
//!
//! [1]: https://www.rust-lang.org
//! "The Rust Programming Language"
//! [2]: https://en.wikipedia.org/wiki/Two-alternative_forced_choice
//! "Wikipedia - Two-alternative forced choice"
//! [3]: https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system
//! "Wikipedia - Hindley-Milner Type System"
//! [4]: https://en.wikipedia.org/wiki/Rewriting#Term_rewriting_systems
//! "Wikipedia - Term Rewriting Systems"
//! [5]: https://en.wikipedia.org/wiki/MU_puzzle "Wikipedia - MU puzzle"
extern crate docopt;
extern crate itertools;
extern crate polytype;
extern crate programinduction;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;
extern crate term_rewriting;
extern crate toml;

use docopt::Docopt;
use itertools::Itertools;
use polytype::Context as TypeContext;
use programinduction::trs::{
    parse_lexicon, parse_rule, parse_trs, task_by_rewrite, GeneticParams, Lexicon, ModelParams, TRS,
};
use programinduction::{GPParams, GP};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fmt;
use std::fs::{read_to_string, File};
use std::io::BufReader;
use std::path::PathBuf;
use std::process::exit;
use term_rewriting::Rule;

fn main() {
    let rng = &mut SmallRng::from_seed([1u8; 16]);

    start_section("Loading parameters");
    let params = exit_err(load_args(), "Problem loading simulation parameters");

    start_section("Loading lexicon");
    let mut lex = exit_err(
        load_lexicon(
            &params.simulation.problem_dir,
            params.simulation.deterministic,
        ),
        "Problem loading lexicon",
    );
    println!("{}", lex);

    start_section("Loading data");
    let data = exit_err(
        load_data(&params.simulation.problem_dir),
        "Problem loading data",
    );
    for datum in &data {
        println!("{}", datum);
    }

    start_section("Loading H*");
    let h_star = exit_err(
        load_h_star(&params.simulation.problem_dir, &mut lex),
        "cannot load H*",
    );
    println!("{}", h_star);

    start_section("Initializing Population");
    let mut pop = exit_err(
        initialize_population(&lex, &params, rng),
        "failed to initialize population",
    );
    for (i, (trs, score)) in pop.iter().enumerate() {
        println!("{}. {}\n{}", i, score, trs);
    }

    start_section("Evolving");
    let evolve_data: Vec<_> = data
        .iter()
        .map(|x| {
            exit_err(
                x.to_positive_example(&mut lex),
                "cannot convert datum to rule",
            )
        })
        .collect();
    exit_err(
        evolve(&evolve_data[..], &mut pop, &h_star, &lex, &params, rng),
        "evolutionary failure",
    );
}

fn start_section(s: &str) {
    println!("\n{}\n{}", s, "-".repeat(s.len()));
}

fn str_err<T, U: ToString>(x: Result<T, U>) -> Result<T, String> {
    x.or_else(|err| Err(err.to_string()))
}

fn exit_err<T>(x: Result<T, String>, msg: &str) -> T {
    x.unwrap_or_else(|err| {
        eprintln!("{}: {}", msg, err);
        exit(1);
    })
}

fn load_args() -> Result<Params, String> {
    let args: Args = Docopt::new("Usage: sim <args-file>")
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());
    let args_file = PathBuf::from(args.arg_args_file);
    let toml_string = str_err(read_to_string(args_file))?;
    str_err(toml::from_str(&toml_string))
}

fn load_lexicon(problem_dir: &str, deterministic: bool) -> Result<Lexicon, String> {
    let sig_path: PathBuf = [problem_dir, "signature"].iter().collect();
    let sig_string = str_err(read_to_string(sig_path))?;
    let bg_path: PathBuf = [problem_dir, "background"].iter().collect();
    let bg_string = str_err(read_to_string(bg_path))?;
    let temp_path: PathBuf = [problem_dir, "templates"].iter().collect();
    let temp_string = str_err(read_to_string(temp_path))?;
    str_err(parse_lexicon(
        &sig_string,
        &bg_string,
        &temp_string,
        deterministic,
        TypeContext::default(),
    ))
}

fn load_data(problem_dir: &str) -> Result<Vec<Record>, String> {
    let path: PathBuf = [problem_dir, "stimuli.json"].iter().collect();
    let file = str_err(File::open(path))?;
    let reader = BufReader::new(file);
    let data: Vec<Record> = str_err(serde_json::from_reader(reader))?;
    Ok(data)
}

fn load_h_star(problem_dir: &str, lex: &mut Lexicon) -> Result<TRS, String> {
    let h_star_file: PathBuf = [problem_dir, "evaluate"].iter().collect();
    let h_star_string = str_err(read_to_string(h_star_file))?;
    str_err(parse_trs(&h_star_string, lex))
}

fn initialize_population<R: Rng>(
    lex: &Lexicon,
    params: &Params,
    rng: &mut R,
) -> Result<Vec<(TRS, f64)>, String> {
    let task = str_err(task_by_rewrite(&[], params.model, lex, ()))?;
    Ok(lex.init(&params.genetic, rng, &params.gp, &task))
}

fn evolve<R: Rng>(
    data: &[Rule],
    pop: &mut Vec<(TRS, f64)>,
    h_star: &TRS,
    lex: &Lexicon,
    params: &Params,
    rng: &mut R,
) -> Result<(), String> {
    println!("n_data,generation,id,llikelihood,lprior,score,difference,description");
    for n_data in 0..=(data.len()) {
        for datum in &data[0..n_data] {
            println!("datum: {}", datum.display());
        }
        let task = str_err(task_by_rewrite(&data[0..n_data], params.model, lex, ()))?;
        for i in pop.iter_mut() {
            i.1 = (task.oracle)(lex, &i.0);
        }
        let h_star_lpost = (task.oracle)(lex, h_star);
        let h_star_llike = -h_star.log_likelihood(&data[0..n_data], params.model);
        let h_star_lprior = -h_star.pseudo_log_prior();

        for gen in 0..params.simulation.generations_per_datum {
            lex.evolve(&params.genetic, rng, &params.gp, &task, pop);
            println!("evolved!");
            for (i, (individual, score)) in pop.iter().enumerate() {
                let llike = -individual.log_likelihood(&data[0..n_data], params.model);
                let lprior = -individual.pseudo_log_prior();
                println!(
                    "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:?}",
                    n_data,
                    gen,
                    i,
                    llike,
                    h_star_llike,
                    lprior,
                    h_star_lprior,
                    score,
                    h_star_lpost,
                    h_star_lpost - score,
                    individual.to_string(),
                );
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    challenge: String,
    correct: String,
    incorrect: String,
    rule: String,
    score: f64,
    stimulus: u16,
}
impl Record {
    fn to_example(&self, rhs: &str, lex: &mut Lexicon) -> Result<Rule, String> {
        let rule_string = format!(
            "S(({})) = S(({}))",
            self.challenge.chars().join(" "),
            rhs.chars().join(" ")
        );
        str_err(parse_rule(&rule_string, lex, &mut lex.context()))
    }
    fn to_positive_example(&self, lex: &mut Lexicon) -> Result<Rule, String> {
        self.to_example(&self.correct, lex)
    }
}
impl fmt::Display for Record {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}, {}, {}, {}→{}, {}↛{}",
            self.stimulus,
            self.rule,
            self.score,
            self.challenge,
            self.correct,
            self.challenge,
            self.incorrect
        )
    }
}

#[derive(Deserialize)]
pub struct Args {
    pub arg_args_file: String,
}

#[derive(Deserialize)]
pub struct Params {
    pub simulation: SimulationParams,
    pub genetic: GeneticParams,
    pub gp: GPParams,
    pub model: ModelParams,
}

#[derive(Serialize, Deserialize)]
pub struct SimulationParams {
    pub generations_per_datum: usize,
    pub problem_dir: String,
    pub deterministic: bool,
}

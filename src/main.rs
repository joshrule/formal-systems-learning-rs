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
use rand::{
    distributions::{Distribution, Uniform},
    seq::SliceRandom,
};
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


    start_section("Scheduling Trials");
    let blocks = exit_err(schedule_trials(&data, 10, 0, &mut lex, rng), "bad data");
    for (i, block) in blocks.iter().enumerate() {
        println!("Block {}: {} trials", i, block.len());
    }

    start_section("Initializing Population");
    let mut pop = exit_err(
        initialize_population(&lex, &params, rng),
        "failed to initialize population",
    );
    for (i, (trs, score)) in pop.iter().enumerate() {
        println!("{}. {}\n{}", i, score, trs);
    }

    start_section("Evolving");
    exit_err(
        evolve(&blocks[0], &mut pop, &h_star, &lex, &params, rng),
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
    data: &[(Rule, Rule)],
    pop: &mut Vec<(TRS, f64)>,
    h_star: &TRS,
    lex: &Lexicon,
    params: &Params,
    rng: &mut R,
) -> Result<(), String> {
    if data.is_empty() {
        return Err(String::from("Not enough data"));
    }
    println!("n_data,generation,id,llikelihood,lprior,score,difference,description");
    for n_data in 0..(data.len() - 1) {
        let test_data = &data[(n_data + 1)..(n_data + 2)];
        for datum in &data[0..(n_data)] {
            println!("datum:\n{}\n{}", datum.0.pretty(), datum.1.pretty());
        }
        let positives = data[0..n_data].iter().map(|(p, _)| p.clone()).collect_vec();
        let task = str_err(task_by_rewrite(&positives, params.model, lex, ()))?;
        for i in pop.iter_mut() {
            i.1 = (task.oracle)(lex, &i.0);
        }
        let h_star_lpost = (task.oracle)(lex, h_star);
        let h_star_llike = -h_star.log_likelihood(&positives, params.model);
        let h_star_lprior = -h_star.pseudo_log_prior();

        for gen in 0..params.simulation.generations_per_datum {
            lex.evolve(&params.genetic, rng, &params.gp, &task, pop);
            for (i, (individual, score)) in pop.iter().enumerate() {
                let llike = -individual.log_likelihood(&positives, params.model);
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
        let prediction = make_2afc_prediction(test_data, params.model, pop);
        println!("prediction: {}", prediction);
    }
    Ok(())
}

fn make_2afc_prediction(
    data: &[(Rule, Rule)],
    params: ModelParams,
    population: &mut Vec<(TRS, f64)>,
) -> f64 {
    let (positives, negatives): (Vec<_>, Vec<_>) = data.iter().cloned().unzip();
    let (ps, ns): (Vec<_>, Vec<_>) = population
        .iter()
        .map(|(t, _)| {
            (
                -t.log_likelihood(&positives, params),
                -t.log_likelihood(&negatives, params),
            )
        })
        .unzip();
    let p = logsumexp(&ps);
    let n = logsumexp(&ns);
    (p - logsumexp(&[p, n])).exp()
}

fn schedule_trials<R: Rng>(
    data: &Vec<Record>,
    n_train: usize,
    n_test: usize,
    lex: &mut Lexicon,
    rng: &mut R,
) -> Result<Vec<Vec<(Rule, Rule)>>, String> {
    let mut blocks = vec![];
    let mut trials = vec![];

    // group trials by rule, shuffle the trials, and shuffle the rule order
    for (_, v) in &data.iter().sorted_by_key(|x| &x.rule).group_by(|x| &x.rule) {
        let mut ts = v.collect_vec();
        ts.shuffle(rng);
        trials.push(ts);
    }
    trials.shuffle(rng);

    for block in 0..(trials.len()) {
        blocks.push(make_training_block(&mut trials, block, n_train, lex, rng)?);
    }

    blocks.push(make_testing_block(&mut trials, n_test, lex, rng)?);

    return Ok(blocks);
}

fn make_training_block<R: Rng>(
    trials: &mut [Vec<&Record>],
    i_block: usize,
    n: usize,
    lex: &mut Lexicon,
    rng: &mut R,
) -> Result<Vec<(Rule, Rule)>, String> {
    let mut block = vec![];
    let blocks = Uniform::new_inclusive(0, i_block);

    // put n-1 trials of current rule in block
    for _ in 0..n {
        // put trial of previous rule in block
        if i_block > 0 {
            let a_previous_rule = blocks.sample(rng);
            // TODO: trial.target = false;
            block.push(trials[a_previous_rule].pop().unwrap().to_examples(lex)?);
        }
        // put trial of current rule in block
        // TODO: trial.target = true;
        block.push(trials[i_block].pop().unwrap().to_examples(lex)?);
    }

    // shuffle, with final trial from current rule
    let last_trial = block.pop().unwrap();
    block.shuffle(rng);
    block.push(last_trial);

    Ok(block)
}

fn make_testing_block<R: Rng>(
    trials: &mut [Vec<&Record>],
    n: usize,
    lex: &mut Lexicon,
    rng: &mut R,
) -> Result<Vec<(Rule, Rule)>, String> {
    trials
        .iter_mut()
        .map(|x| x.drain(..n))
        .flatten()
        .map(|x| x.to_examples(lex))
        .collect::<Result<Vec<_>, String>>()
        .map(|mut x| {
            x.shuffle(rng);
            x
        })
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
            "S(({} END{})) = S(({} END{}))",
            self.challenge.chars().join(" ("),
            ")".repeat(self.challenge.len() - 1),
            rhs.chars().join(" ("),
            ")".repeat(rhs.len() - 1),
        );
        str_err(parse_rule(&rule_string, lex, &mut lex.context()))
    }
    fn to_positive_example(&self, lex: &mut Lexicon) -> Result<Rule, String> {
        self.to_example(&self.correct, lex)
    }
    fn to_negative_example(&self, lex: &mut Lexicon) -> Result<Rule, String> {
        self.to_example(&self.incorrect, lex)
    }
    fn to_examples(&self, lex: &mut Lexicon) -> Result<(Rule, Rule), String> {
        Ok((
            self.to_positive_example(lex)?,
            self.to_negative_example(lex)?,
        ))
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

pub fn logsumexp(lps: &[f64]) -> f64 {
    let largest = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let x = lps.iter().map(|lp| (lp - largest).exp()).sum::<f64>().ln();
    largest + x
}

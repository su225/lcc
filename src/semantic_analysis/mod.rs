pub mod identifier_resolution;
pub mod loop_labeling;
pub mod typechecking;

#[cfg(test)]
mod desugaring_verifier;

#[cfg(test)]
mod unique_identifier_verifier;

#[cfg(test)]
mod loop_label_verifier;

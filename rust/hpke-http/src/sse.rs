//! Byte-level SSE block splitting. Field parsing belongs to the caller.

use crate::Error;

/// One bounded partial SSE block. Line endings on the wire are LF.
#[derive(Debug)]
pub struct SseSplitter {
    pending: Vec<u8>,
    line_has_data: bool,
    skip_lf: bool,
    max_block_len: usize,
    closed: bool,
}

impl SseSplitter {
    /// Start a splitter with the same limit used for a response DATA record.
    #[must_use]
    pub fn new(max_block_len: usize) -> Self {
        Self {
            pending: Vec::new(),
            line_has_data: false,
            skip_lf: false,
            max_block_len,
            closed: false,
        }
    }

    /// Read at most one complete block and report the number of input bytes used.
    /// Pass the unread suffix to the next call.
    ///
    /// # Errors
    /// Returns [`Error::LimitExceeded`] if one block exceeds its bound, or
    /// [`Error::InvalidConfiguration`] if the splitter is closed.
    pub fn feed(&mut self, input: &[u8]) -> Result<(usize, Option<Vec<u8>>), Error> {
        if self.closed {
            return Err(Error::InvalidConfiguration);
        }
        for (index, &byte) in input.iter().enumerate() {
            if self.skip_lf {
                self.skip_lf = false;
                if byte == b'\n' {
                    continue;
                }
            }
            if byte == b'\r' || byte == b'\n' {
                if byte == b'\r' {
                    self.skip_lf = true;
                }
                self.push(b'\n')?;
                if !self.line_has_data {
                    return Ok((index + 1, Some(std::mem::take(&mut self.pending))));
                }
                self.line_has_data = false;
            } else {
                self.push(byte)?;
                self.line_has_data = true;
            }
        }
        Ok((input.len(), None))
    }

    /// Discard an incomplete final block and close the splitter.
    pub fn finish(&mut self) {
        self.pending.clear();
        self.closed = true;
    }

    fn push(&mut self, byte: u8) -> Result<(), Error> {
        if self.pending.len() >= self.max_block_len {
            self.closed = true;
            self.pending.clear();
            return Err(Error::LimitExceeded);
        }
        self.pending.push(byte);
        Ok(())
    }
}

pub(crate) fn validate_block(block: &[u8], max_block_len: usize) -> Result<(), Error> {
    if block.len() > max_block_len {
        return Err(Error::LimitExceeded);
    }
    if block.is_empty() || block.contains(&b'\r') || !block.ends_with(b"\n") {
        return Err(Error::MalformedEnvelope);
    }
    let mut line_start = 0;
    for (index, &byte) in block.iter().enumerate() {
        if byte == b'\n' {
            if index == line_start {
                return if index + 1 == block.len() {
                    Ok(())
                } else {
                    Err(Error::MalformedEnvelope)
                };
            }
            line_start = index + 1;
        }
    }
    Err(Error::MalformedEnvelope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_each_line_ending_without_losing_bytes() -> Result<(), Error> {
        let mut splitter = SseSplitter::new(64);
        let mut output = Vec::new();
        for byte in b"data:\r\n\rid: 1\n\n" {
            let (used, block) = splitter.feed(&[*byte])?;
            assert_eq!(used, 1);
            if let Some(block) = block {
                output.push(block);
            }
        }
        assert_eq!(output, vec![b"data:\n\n".to_vec(), b"id: 1\n\n".to_vec()]);
        Ok(())
    }

    #[test]
    fn keeps_raw_non_line_bytes_and_drops_tail() -> Result<(), Error> {
        let mut splitter = SseSplitter::new(64);
        let input = b"\xef\xbb\xbfdata:\xff\n\npartial";
        let (used, block) = splitter.feed(input)?;
        assert_eq!(used, 11);
        assert_eq!(
            block.ok_or(Error::MalformedEnvelope)?,
            b"\xef\xbb\xbfdata:\xff\n\n"
        );
        splitter.feed(&input[used..])?;
        splitter.finish();
        assert!(splitter.feed(b"\n").is_err());
        Ok(())
    }

    #[test]
    fn rejects_non_single_blocks() {
        assert!(validate_block(b"\n", 1).is_ok());
        for value in [b"data:x\n".as_slice(), b"data:x\n\n\n", b"data:x\r\n\r\n"] {
            assert_eq!(validate_block(value, 100), Err(Error::MalformedEnvelope));
        }
    }

    #[test]
    fn exact_block_limit_passes_and_one_more_byte_fails() -> Result<(), Error> {
        let mut exact = SseSplitter::new(3);
        assert_eq!(exact.feed(b"x\n\n")?, (3, Some(b"x\n\n".to_vec())));
        let mut short = SseSplitter::new(2);
        assert_eq!(short.feed(b"x\n\n"), Err(Error::LimitExceeded));
        assert!(short.feed(b"\n").is_err());
        Ok(())
    }
}

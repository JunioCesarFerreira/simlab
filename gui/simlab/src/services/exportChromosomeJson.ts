import type {
  ChromosomeDraft,
  ExportedChromosomeFile,
  MacProtocol,
  ProblemName,
} from '../types/problem'

function macGene(mac: MacProtocol): 0 | 1 {
  return mac === 'tsch' ? 1 : 0
}

export function exportChromosome(chrom: ChromosomeDraft): ExportedChromosomeFile {
  const problem_name = chrom.kind as ProblemName

  if (chrom.kind === 'problem1') {
    return {
      problem_name,
      chromosome: {
        mac_protocol: macGene(chrom.macProtocol),
        relays: chrom.relays.map(r => ({ x: r.x, y: r.y })),
      },
    }
  }

  if (chrom.kind === 'problem2' || chrom.kind === 'problem3') {
    return {
      problem_name,
      chromosome: {
        mac_protocol: macGene(chrom.macProtocol),
        mask: [...chrom.mask],
      },
    }
  }

  return {
    problem_name,
    chromosome: {
      mac_protocol: macGene(chrom.macProtocol),
      route: [...chrom.route],
      sojourn_times: [...chrom.sojournTimes],
    },
  }
}

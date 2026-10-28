// Expand external assets where they execute so source-level tests retain ordering.
import {readFileSync} from 'node:fs';
export function benchmarkSource() {
  const root=new URL('../',import.meta.url);
  return readFileSync(new URL('templates/index.html',root),'utf8')
    .replace(/<script src="\{\{ benchmark_asset\('([^']+)'\) \}\}"><\/script>/g,(_tag,name)=>
      '<script>/* '+name+' */'+readFileSync(new URL((name.endsWith('.html')?'templates/':'static/')+name,root),'utf8')+'</script>')
    .replace(/<link rel="stylesheet" href="\{\{ benchmark_asset\('benchmark.css'\) \}\}">/,
      '<style>'+readFileSync(new URL('static/benchmark.css',root),'utf8')+'</style>');
}

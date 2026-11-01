export default function StreamLoader() {
  return (
    <div className="stream-loader" role="status" aria-label="Receiving response">
      <span className="stream-loader-orbit" aria-hidden="true">
        <i />
      </span>
      <span>Working</span>
      <span className="stream-loader-dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
    </div>
  )
}
